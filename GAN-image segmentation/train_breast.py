import os
import logging
from pathlib import Path
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from unet.unet_model import UNetWithDeepSupervision
from unet.discriminator import Discriminator
from utils.data_loading import BasicDatasetFromPaths
from utils.dice_score import dice_loss
from utils.gan_seg_diagnostics import evaluate_model, dice_threshold_curve, size_stratified_metrics, area_ks_test
from utils.loss_recorder import TrainingPlotter


img_dir = "data/train"
mask_dir = "data/mask"
img_aug_dir = "data/train_strong"
mask_aug_dir = "data/mask_strong"

dir_checkpoint = Path('./checkpoints/breast/')

# ------------------
# 训练超参
# ------------------
epochs = 200
batch_size = 8
lr_g = 1e-4
lr_d = 1e-4
lambda_adv = 0.01             # 保持你原有的对抗权重
aux_weights = [0.4, 0.2, 0.1] # 深监督各尺度权重（由近到远）
img_scale = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_augmented_paths(original_paths, aug_img_dir, aug_mask_dir):
    selected_aug_imgs, selected_aug_masks = [], []
    base_names = [os.path.splitext(os.path.basename(p))[0] for p in original_paths]
    for name in base_names:
        selected_aug_imgs.extend(glob(os.path.join(aug_img_dir, f"{name}_*.png")))
        selected_aug_masks.extend(glob(os.path.join(aug_mask_dir, f"{name}_*.png")))
    return selected_aug_imgs, selected_aug_masks


def get_dataloaders():
    img_paths = sorted(glob(os.path.join(img_dir, "*.png")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))


    val_count = 7
    train_img_ori, val_img_ori, train_mask_ori, val_mask_ori = train_test_split(
        img_paths, mask_paths, test_size=val_count, random_state=42
    )


    train_img_aug, train_mask_aug = get_augmented_paths(train_img_ori, img_aug_dir, mask_aug_dir)
    final_train_imgs = train_img_ori + train_img_aug
    final_train_masks = train_mask_ori + train_mask_aug

    train_set = BasicDatasetFromPaths(final_train_imgs, final_train_masks, img_scale)
    val_set = BasicDatasetFromPaths(val_img_ori, val_mask_ori, img_scale)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            drop_last=False, num_workers=4, pin_memory=True)
    print(f"训练集: {len(final_train_imgs)} (含增强 {len(train_img_aug)}) | 验证集: {len(val_img_ori)}")
    return train_loader, val_loader


def evaluate_simple(model, loader):
    model.eval()
    dices, ious, recalls = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, dtype=torch.float32)
            gts = batch['mask'].to(device, dtype=torch.float32)  # [B,H,W] or [B,1,H,W]
            if gts.dim() == 4:
                gts = gts.squeeze(1)
            logits = model(images)
            prob = torch.sigmoid(logits).squeeze(1)
            pred = (prob > 0.5).float()

            inter = (pred * gts).sum(dim=(1,2))
            dice = (2*inter + 1e-6)/ (pred.sum(dim=(1,2)) + gts.sum(dim=(1,2)) + 1e-6)
            dices.extend(dice.cpu().numpy())

            union = pred + gts
            iou = (inter + 1e-6)/ (union.sum(dim=(1,2)) - inter + 1e-6)
            ious.extend(iou.cpu().numpy())

            tp = (pred * gts).sum(dim=(1,2))
            fn = ((1-pred) * gts).sum(dim=(1,2))
            rec = (tp + 1e-6)/ (tp + fn + 1e-6)
            recalls.extend(rec.cpu().numpy())

    return np.mean(dices), np.mean(ious), np.mean(recalls)


def train():
    plotter = TrainingPlotter(save_dir="./checkpoints")

    G = UNetWithDeepSupervision(in_channels=3, out_channels=1, n_aux=3).to(device)
    D = Discriminator(in_channels=4).to(device)

    for name, p in G.encoder.named_parameters():
        p.requires_grad = False
    for name, p in G.encoder.named_parameters():
        p.requires_grad = ("layer4" in name)

    optimizer_g = optim.Adam(G.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(D.parameters(), lr=lr_d)

    bce = nn.BCELoss()
    seg_criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader = get_dataloaders()

    best_metrics = {'dice': 0, 'iou': 0, 'recall': 0, 'epoch': 0}

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_dice = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device, dtype=torch.float32)          # [B,3,H,W]
                gts = batch['mask'].to(device, dtype=torch.float32)              # [B,H,W]
                gts4 = gts.unsqueeze(1)                                          # [B,1,H,W]

                # -------- Train D --------
                optimizer_d.zero_grad()
                with torch.no_grad():
                    main_fake = G(images)
                real_pred = D(images, gts4)
                fake_pred = D(images, main_fake.detach())
                d_real = bce(real_pred, torch.ones_like(real_pred))
                d_fake = bce(fake_pred, torch.zeros_like(fake_pred))
                d_loss = 0.5 * (d_real + d_fake)
                d_loss.backward()
                optimizer_d.step()

                # -------- Train G --------
                optimizer_g.zero_grad()
                main_logits, aux_logits_list = G.forward_with_aux(images)


                loss_main = seg_criterion(main_logits.squeeze(1), gts) + \
                            dice_loss(torch.sigmoid(main_logits.squeeze(1)), gts, multiclass=False)


                loss_aux_total = 0.0
                for w, aux_logits in zip(aux_weights, aux_logits_list):
                    loss_aux = seg_criterion(aux_logits.squeeze(1), gts) + \
                               dice_loss(torch.sigmoid(aux_logits.squeeze(1)), gts, multiclass=False)
                    loss_aux_total += w * loss_aux


                fake_for_g = D(images, main_logits)
                loss_adv = bce(fake_for_g, torch.ones_like(fake_for_g))

                g_loss = loss_main + loss_aux_total + lambda_adv * loss_adv
                g_loss.backward()
                optimizer_g.step()


                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                with torch.no_grad():
                    cur_dice = 1.0 - dice_loss(torch.sigmoid(main_logits.squeeze(1)), gts, multiclass=False)
                epoch_dice += cur_dice.item()

                pbar.update(1)
                pbar.set_postfix({
                    'G': f"{(epoch_g_loss/pbar.n):.4f}",
                    'D': f"{(epoch_d_loss/pbar.n):.4f}",
                    'Main': f"{loss_main.item():.3f}",
                    'Aux': f"{loss_aux_total.item():.3f}",
                    'Adv': f"{(lambda_adv*loss_adv).item():.3f}",
                    'Dice': f"{cur_dice.item():.3f}",
                })

        print(f"G_loss: {epoch_g_loss/len(train_loader):.4f} | "
              f"D_loss: {epoch_d_loss/len(train_loader):.4f} | "
              f"Train Dice: {epoch_dice/len(train_loader):.4f}")


        val_metrics = evaluate_model(G, val_loader, device, threshold=0.5)
        print(f"Validation - Dice: {val_metrics['Dice']:.4f}, "
              f"IoU: {val_metrics['IoU']:.4f}, Recall: {val_metrics['Recall']:.4f}")


        plotter.update(
            g_loss=epoch_g_loss / len(train_loader),
            d_loss=epoch_d_loss / len(train_loader),
            train_dice=epoch_dice / len(train_loader),
            val_dice=val_metrics['Dice'],
            val_iou=val_metrics['IoU'],
            val_recall=val_metrics['Recall']
        )


        if val_metrics['Dice'] > best_metrics['dice'] and val_metrics['IoU'] > best_metrics['iou']:
            best_metrics = {
                'dice': val_metrics['Dice'],
                'iou': val_metrics['IoU'],
                'recall': val_metrics['Recall'],
                'epoch': epoch
            }
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(G.state_dict(), str(dir_checkpoint / 'best_G_deepsup_breast.pth'))
            torch.save(D.state_dict(), str(dir_checkpoint / 'best_D_deepsup_breast.pth'))
            print(f"✅ 模型已更新 (Epoch {epoch})")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(f"Using device: {device}")
    train()
