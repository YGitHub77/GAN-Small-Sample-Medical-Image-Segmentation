
import os
from pathlib import Path
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from inspect import signature

# ========== 请根据你的工程结构调整导入路径 ==========
from unet.unet_model import UNetWithDeepSupervision
from utils.data_loading import BasicDatasetFromPaths


FOLDS = 5
SEED = 42
IMG_SCALE = 0.5
BATCH_SIZE = 1
THRESHOLD = 0.5

IMG_DIR = Path(r"data\Lung\image")
MASK_DIR = Path(r"data\Lung\mask")

WEIGHTS_PATTERN = r"fold{fold}\G_best.pth"

OUT_DIR = Path(r"")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize_bchw(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4, f"expect BCHW, got {tuple(x.shape)}"
    if torch.nan_to_num(x.max()) > 2.0:
        x = x / 255.0
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, normalize_fn=imagenet_normalize_bchw):
        self.base = base_ds
        self.normalize_fn = normalize_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item['image']
        if img.ndim == 3:
            img = img.unsqueeze(0)
            img = self.normalize_fn(img).squeeze(0)
        else:
            img = self.normalize_fn(img)
        item['image'] = img
        return item


def binarize_mask(prob_map, thr=0.5):
    return (prob_map >= thr).astype(np.uint8)


def compute_confusion(pred_bin, gt_bin):
    tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = int(((pred_bin == 0) & (gt_bin == 0)).sum())
    return tp, fp, fn, tn


def dice_from_conf(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


def iou_from_conf(tp, fp, fn):
    denom = tp + fp + fn
    return (tp / denom) if denom > 0 else 1.0


def recall_from_conf(tp, fn):
    denom = tp + fn
    return (tp / denom) if denom > 0 else 1.0


def robust_model_forward(model, image):
    if hasattr(model, "forward_with_aux"):
        try:
            out = model.forward_with_aux(image)
            if isinstance(out, (tuple, list)):
                return out[0]
            return out
        except Exception:
            pass

    try:
        out = model(image)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out
    except TypeError as e_main:
        alt_attrs = ['model', 'net', 'module']
        for a in alt_attrs:
            if hasattr(model, a):
                m2 = getattr(model, a)
                try:
                    out = m2(image)
                    if isinstance(out, (tuple, list)):
                        return out[0]
                    return out
                except Exception:
                    pass

        if hasattr(model, 'forward'):
            try:
                out = model.forward(image)
                if isinstance(out, (tuple, list)):
                    return out[0]
                return out
            except Exception:
                pass

        enc_candidates = []
        if hasattr(model, 'encoder'):
            enc_candidates.append(model.encoder)
        if hasattr(model, 'net') and hasattr(model.net, 'encoder'):
            enc_candidates.append(model.net.encoder)
        if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            enc_candidates.append(model.model.encoder)

        for enc in enc_candidates:
            try:
                feats = enc(image)
                if isinstance(feats, torch.Tensor):
                    if hasattr(model, 'segmentation_head'):
                        try:
                            return model.segmentation_head(feats)
                        except Exception:
                            pass
                    if hasattr(model, 'net') and hasattr(model.net, 'segmentation_head'):
                        try:
                            return model.net.segmentation_head(feats)
                        except Exception:
                            pass
                if isinstance(feats, (tuple, list)):
                    dec_candidates = []
                    if hasattr(model, 'decoder'):
                        dec_candidates.append(model.decoder)
                    if hasattr(model, 'net') and hasattr(model.net, 'decoder'):
                        dec_candidates.append(model.net.decoder)
                    if hasattr(model, 'model') and hasattr(model.model, 'decoder'):
                        dec_candidates.append(model.model.decoder)

                    for dec in dec_candidates:
                        try:
                            out = dec(feats)
                            if isinstance(out, torch.Tensor):
                                if hasattr(model, 'segmentation_head'):
                                    try:
                                        return model.segmentation_head(out)
                                    except Exception:
                                        return out
                                return out
                        except Exception:
                            pass
                        try:
                            out = dec(*feats)
                            if isinstance(out, torch.Tensor):
                                if hasattr(model, 'segmentation_head'):
                                    try:
                                        return model.segmentation_head(out)
                                    except Exception:
                                        return out
                                return out
                        except Exception:
                            pass
            except Exception:
                pass

        print("=== model forward failed. Debug info follow ===")
        try:
            print("Model class:", model.__class__)
            print("Signature of model.forward():", signature(model.forward))
        except Exception:
            print("Cannot inspect signature.")
        print("Model repr:")
        print(model)
        raise e_main


def main():
    img_paths = sorted(glob(str(IMG_DIR / "*.png")))
    mask_paths = sorted(glob(str(MASK_DIR / "*.png")))
    assert len(img_paths) == len(mask_paths), "Image / mask count mismatch."
    n_samples = len(img_paths)
    print(f"Total samples found: {n_samples}")

    kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(img_paths))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        fold_no = fold_idx + 1
        print(f"\n=== Predicting fold {fold_no}/{FOLDS} ===")
        weight_path = WEIGHTS_PATTERN.format(fold=fold_no)
        if not Path(weight_path).exists():
            print(f"[WARN] Weight file not found for fold {fold_no}: {weight_path}")
            continue


        model = UNetWithDeepSupervision(in_channels=3, out_channels=1, n_aux=3)
        model = model.to(DEVICE)
        ckpt = torch.load(weight_path, map_location=DEVICE)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
            state = {k.replace("module.", ""): v for k, v in ckpt.items()}
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            state = None

        if state is not None:
            try:
                model.load_state_dict(state)
            except RuntimeError:
                model.load_state_dict(state, strict=False)
        else:
            try:
                model.load_state_dict(ckpt)
            except Exception as e:
                print(f"Failed to load checkpoint for fold {fold_no}: {e}")
                continue

        model.eval()


        val_img_paths = [img_paths[i] for i in va_idx]
        val_mask_paths = [mask_paths[i] for i in va_idx]
        base_val_set = BasicDatasetFromPaths(val_img_paths, val_mask_paths, IMG_SCALE)
        val_set = NormalizedDataset(base_val_set)  # 应用训练时的归一化
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


        merged_txt = OUT_DIR / f"fold{fold_no}_metrics.txt"

        per_image_lines = []
        dices = []
        ious = []
        recalls = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Fold {fold_no} predicting", unit="img"):
                if isinstance(batch, dict):
                    image = batch.get('image')
                    mask = batch.get('mask')
                    name = batch.get('name') or batch.get('img_name') or None
                else:
                    if len(batch) >= 2:
                        image = batch[0]
                        mask = batch[1]
                        name = batch[2] if len(batch) > 2 else None
                    else:
                        raise RuntimeError("Unexpected batch format from dataset")

                image = image.to(DEVICE, dtype=torch.float32)
                output = robust_model_forward(model, image)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                probs = torch.sigmoid(output).cpu().numpy()

                gt = mask.cpu().numpy()
                if probs.ndim == 4 and probs.shape[1] == 1:
                    probs = probs[:, 0, :, :]
                if gt.ndim == 4 and gt.shape[1] == 1:
                    gt = gt[:, 0, :, :]

                for b in range(probs.shape[0]):
                    prob_map = probs[b]
                    gt_map = gt[b]
                    pred_bin = binarize_mask(prob_map, thr=THRESHOLD)
                    gt_bin = (gt_map >= 0.5).astype(np.uint8)

                    tp, fp, fn, tn = compute_confusion(pred_bin, gt_bin)
                    dice = dice_from_conf(tp, fp, fn)
                    iou = iou_from_conf(tp, fp, fn)
                    recall = recall_from_conf(tp, fn)

                    dices.append(dice)
                    ious.append(iou)
                    recalls.append(recall)


                    im_name = None
                    if isinstance(name, (list, tuple)) and len(name) > b:
                        im_name = name[b]
                    elif isinstance(name, (str, bytes)):
                        im_name = name
                    else:
                        try:
                            im_name = Path(val_img_paths[len(dices) - 1]).name
                        except Exception:
                            im_name = f"sample_{len(dices)}"

                    try:
                        im_name = im_name.decode() if isinstance(im_name, bytes) else str(im_name)
                    except Exception:
                        im_name = f"sample_{len(dices)}"

                    per_image_lines.append(f"{im_name}\t{dice:.6f}\t{iou:.6f}\t{recall:.6f}")


        mean_dice = float(np.mean(dices)) if len(dices) > 0 else 0.0
        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        mean_recall = float(np.mean(recalls)) if len(recalls) > 0 else 0.0


        with open(merged_txt, "w", encoding="utf-8") as f:
            # 汇总部分
            f.write("=" * 50 + "\n")
            f.write(f"Fold {fold_no} 汇总指标\n")
            f.write("=" * 50 + "\n")
            f.write(f"权重文件路径: {weight_path}\n")
            f.write(f"验证集图像数量: {len(dices)}\n")
            f.write(f"平均Dice系数: {mean_dice:.6f}\n")
            f.write(f"平均IoU: {mean_iou:.6f}\n")
            f.write(f"平均Recall: {mean_recall:.6f}\n")
            f.write(f"使用阈值: {THRESHOLD}\n")
            f.write("\n" + "=" * 50 + "\n")

            # 单图像详细指标部分
            f.write("单图像详细指标（图像名称\tDice\tIoU\tRecall）\n")
            f.write("=" * 50 + "\n")
            f.write("\n".join(per_image_lines))

        print(f"Fold {fold_no} done. 结果已保存至: {merged_txt}")

    print("\nAll folds done.")


if __name__ == "__main__":
    main()