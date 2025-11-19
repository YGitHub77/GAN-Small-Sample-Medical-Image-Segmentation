import os
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from unet.unet_model import UNetWithDeepSupervision
import segmentation_models_pytorch as smp

from unet.discriminator import Discriminator
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.gan_seg_diagnostics import (
    evaluate_model, dice_threshold_curve, size_stratified_metrics, area_ks_test
)
from utils.loss_recorder import TrainingPlotter


SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_DIR = SCRIPT_DIR / "checkpoints/lung"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

dir_img = Path('data/Lung/image')
dir_mask = Path('data/Lung/mask')

print(f"[PATH] CWD        = {os.getcwd()}")
print(f"[PATH] SCRIPT_DIR = {SCRIPT_DIR}")
print(f"[PATH] CKPT_DIR   = {CKPT_DIR}  <-- 权重固定存这里")


epochs = 200
batch_size = 16
lr_g = 1e-4
lr_d = 1e-4
lambda_adv = 0.01
aux_weights = [0.3, 0.15, 0.05]
img_scale = 1
val_percent = 0.1
seed = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(seed)



def build_dataloaders():
    # 兼容 CarvanaDataset / BasicDataset（与肺部脚本一致的接口）
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, scale=img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, scale=img_scale)

    n_val = max(1, int(len(dataset) * val_percent))
    n_train = len(dataset) - n_val
    # 用 42 做划分种子
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    print(f"Train={len(train_set)} | Val={len(val_set)} | Total={len(dataset)} | Seed={seed}")
    return train_loader, val_loader



def train():
    plotter = TrainingPlotter(save_dir=str(CKPT_DIR))

    G = UNetWithDeepSupervision(in_channels=3, out_channels=1, n_aux=len(aux_weights)).to(device)

    D = Discriminator(in_channels=4).to(device)  # 3通道图 + 1通道logits

    for _, p in G.encoder.named_parameters():
        p.requires_grad = False
    for name, p in G.encoder.named_parameters():
        p.requires_grad = ("layer4" in name)

    optimizer_g = optim.Adam(G.parameters(), lr=lr_g)
    optimizer_d = optim.Adam(D.parameters(), lr=lr_d)

    bce = nn.BCELoss()
    seg_criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader = build_dataloaders()

    best = {'dice': 0.0, 'iou': 0.0, 'recall': 0.0, 'epoch': 0}

    print(f"Trainable params: {sum(p.numel() for p in G.parameters() if p.requires_grad)}")
    print(f"Total params:     {sum(p.numel() for p in G.parameters())}")

    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_dice = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device, dtype=torch.float32)    # [B,3,H,W]
                gts    = batch['mask'].to(device, dtype=torch.float32)     # [B,H,W]
                gts4   = gts.unsqueeze(1)                                   # [B,1,H,W]

                # ---- Train D ----
                optimizer_d.zero_grad()
                with torch.no_grad():
                    main_fake = G(images)
                real_pred = D(images, gts4)
                fake_pred = D(images, main_fake.detach())
                d_loss = 0.5 * (bce(real_pred, torch.ones_like(real_pred)) +
                                bce(fake_pred, torch.zeros_like(fake_pred)))
                d_loss.backward()
                optimizer_d.step()

                # ---- Train G ----
                optimizer_g.zero_grad()
                main_logits, aux_list = G.forward_with_aux(images)


                loss_main = seg_criterion(main_logits.squeeze(1), gts) + \
                            dice_loss(torch.sigmoid(main_logits.squeeze(1)), gts, multiclass=False)


                loss_aux_total = 0.0
                for w, aux_logits in zip(aux_weights, aux_list):
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
        print(f"Validation - Dice: {val_metrics['Dice']:.4f}, IoU: {val_metrics['IoU']:.4f}, "
              f"Recall: {val_metrics['Recall']:.4f}")


        plotter.update(
            g_loss=epoch_g_loss / len(train_loader),
            d_loss=epoch_d_loss / len(train_loader),
            train_dice=epoch_dice / len(train_loader),
            val_dice=val_metrics['Dice'],
            val_iou=val_metrics['IoU'],
            val_recall=val_metrics['Recall']
        )


        is_best = (val_metrics['Dice'] > best['dice']) and (val_metrics['IoU'] > best['iou'])
        if is_best:
            best = {
                'dice':   val_metrics['Dice'],
                'iou':    val_metrics['IoU'],
                'recall': val_metrics['Recall'],
                'epoch':  epoch
            }
            g_path = (CKPT_DIR / 'best_G_deepsup_lung.pth').resolve()
            d_path = (CKPT_DIR / 'best_D_deepsup_lung.pth').resolve()
            torch.save(G.state_dict(), str(g_path))
            torch.save(D.state_dict(), str(d_path))
            print(f"✅ 模型已更新 (Epoch {epoch})")



    print(f"\n最佳 Epoch {best['epoch']} | Dice={best['dice']:.4f}, IoU={best['iou']:.4f}, Recall={best['recall']:.4f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(f"Using device: {device}")
    train()
