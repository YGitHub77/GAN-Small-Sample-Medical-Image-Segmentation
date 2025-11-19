"""
gan_seg_diagnostics.py (兼容 3D/4D mask)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F

try:
    import cv2
except Exception:
    cv2 = None
try:
    from scipy import ndimage as ndi
    from scipy.stats import ks_2samp
except Exception:
    ndi = None
    ks_2samp = None


def _pick_gt(gts_np: np.ndarray, i: int) -> np.ndarray:
    """
    支持 gts shape 为 (N,1,H,W) 或 (N,H,W)，返回 (H,W) 的 0/1 numpy mask
    """
    if gts_np.ndim == 4:  # (N,1,H,W)
        return (gts_np[i, 0] > 0.5).astype(np.uint8)
    elif gts_np.ndim == 3:  # (N,H,W)
        return (gts_np[i] > 0.5).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected GT shape {gts_np.shape}")


def binarize(prob: np.ndarray, thr: float) -> np.ndarray:
    return (prob >= thr).astype(np.uint8)


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return (2.0 * inter + eps) / (denom + eps)


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)


def precision_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp = (pred & gt).sum()
    fp = (pred & (~gt.astype(bool))).sum()
    return (tp + eps) / (tp + fp + eps)


def recall_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp = (pred & gt).sum()
    fn = ((~pred.astype(bool)) & gt.astype(bool)).sum()
    return (tp + eps) / (tp + fn + eps)


def extract_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    mask_bin = (mask > 0).astype(np.uint8)
    if cv2 is not None:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask_bin, kernel, iterations=thickness)
    elif ndi is not None:
        eroded = ndi.binary_erosion(mask_bin, iterations=thickness).astype(np.uint8)
    else:
        eroded = mask_bin
    boundary = mask_bin - eroded
    boundary[boundary < 0] = 0
    return boundary.astype(np.uint8)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, delta: int = 2) -> float:
    if ndi is None:
        return 0.0
    pred_b = extract_boundary(pred)
    gt_b = extract_boundary(gt)
    gt_dt = ndi.distance_transform_edt(1 - gt_b)
    pred_dt = ndi.distance_transform_edt(1 - pred_b)
    pred_match = (pred_b > 0) & (gt_dt <= delta)
    gt_match = (gt_b > 0) & (pred_dt <= delta)
    tp = pred_match.sum()
    fp = (pred_b > 0).sum() - tp
    fn = (gt_b > 0).sum() - gt_match.sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / (prec + rec + 1e-7)


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if ndi is None:
        return np.array([0.0])
    pred_b = extract_boundary(pred).astype(bool)
    gt_b = extract_boundary(gt).astype(bool)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return np.array([0.0])
    gt_dt = ndi.distance_transform_edt(~gt_b)
    pred_dt = ndi.distance_transform_edt(~pred_b)
    d_pred_to_gt = gt_dt[pred_b]
    d_gt_to_pred = pred_dt[gt_b]
    return np.concatenate([d_pred_to_gt, d_gt_to_pred]).astype(np.float32)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    dists = _surface_distances(pred, gt)
    return float(np.percentile(dists, 95)) if dists.size else 0.0


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    dists = _surface_distances(pred, gt)
    return float(dists.mean()) if dists.size else 0.0


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   loader,
                   device: torch.device,
                   threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    dices, ious, precs, recs, hd95s, assds, bf1s = [], [], [], [], [], [], []
    for batch in loader:
        imgs, gts = batch['image'].to(device), batch['mask']
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        gts_np = gts.cpu().numpy()
        for i in range(probs.shape[0]):
            prob = probs[i, 0]
            gt = _pick_gt(gts_np, i)
            pred = binarize(prob, threshold)
            dices.append(dice_score(pred, gt))
            ious.append(iou_score(pred, gt))
            precs.append(precision_score(pred, gt))
            recs.append(recall_score(pred, gt))
            hd95s.append(hd95(pred, gt))
            assds.append(assd(pred, gt))
            bf1s.append(boundary_f1(pred, gt, delta=2))
    return {
        "Dice": float(np.mean(dices)),
        "IoU": float(np.mean(ious)),
        "Precision": float(np.mean(precs)),
        "Recall": float(np.mean(recs)),
        "HD95": float(np.mean(hd95s)),
        "ASSD": float(np.mean(assds)),
        "BF1": float(np.mean(bf1s)),
    }


@torch.no_grad()
def dice_threshold_curve(model, loader, device, thr_list=None):
    if thr_list is None:
        thr_list = list(np.linspace(0.3, 0.7, 21))
    model.eval()
    accum = np.zeros(len(thr_list))
    count = 0
    for batch in loader:
        imgs, gts = batch['image'].to(device), batch['mask']
        probs = torch.sigmoid(model(imgs)).cpu().numpy()
        gts_np = gts.cpu().numpy()
        for i in range(probs.shape[0]):
            prob = probs[i, 0]
            gt = _pick_gt(gts_np, i)
            for k, t in enumerate(thr_list):
                accum[k] += dice_score(binarize(prob, t), gt)
            count += 1
    return {"thr": np.array(thr_list), "dice": accum / max(count, 1)}

import numpy as np
import torch

def size_stratified_metrics(G, val_loader, device, threshold=0.5):
    """
    按 GT 掩膜像素面积三分位划分 small/medium/large，
    对每个尺寸桶分别统计 Dice 与 IoU 的均值/方差等。
    返回示例：
    {
        'small':  {'count': n, 'dice_mean': x, 'dice_std': y, 'iou_mean': a, 'iou_std': b},
        'medium': {...},
        'large':  {...},
        'cuts':   {'p33': v1, 'p66': v2}   # 便于复现实验
    }
    """
    G.eval()
    probs_list, gts_list, areas = [], [], []

    def _dice_np(p, g):
        inter = np.logical_and(p == 1, g == 1).sum()
        s = p.sum() + g.sum()
        return 1.0 if s == 0 else (2.0 * inter) / (s + 1e-8)

    def _iou_np(p, g):
        inter = np.logical_and(p == 1, g == 1).sum()
        union = np.logical_or(p == 1, g == 1).sum()
        return 1.0 if union == 0 else inter / (union + 1e-8)


    with torch.no_grad():
        for batch in val_loader:
            imgs, gts = batch['image'].to(device), batch['mask']
            prob = torch.sigmoid(G(imgs)).cpu().numpy()     # [B,1,H,W]
            gts_np = gts.numpy()                             # [B,1,H,W] 或 [B,H,W]
            bsz = prob.shape[0]
            for i in range(bsz):
                p = prob[i, 0]  # HxW, float
                g = (gts_np[i] if gts_np.ndim == 3 else gts_np[i, 0] > 0.5).astype(np.uint8)
                probs_list.append(p)
                gts_list.append(g)
                areas.append(int(g.sum()))

    if len(areas) == 0:
        return {
            'small':  {'count': 0, 'dice_mean': 0.0, 'dice_std': 0.0, 'iou_mean': 0.0, 'iou_std': 0.0},
            'medium': {'count': 0, 'dice_mean': 0.0, 'dice_std': 0.0, 'iou_mean': 0.0, 'iou_std': 0.0},
            'large':  {'count': 0, 'dice_mean': 0.0, 'dice_std': 0.0, 'iou_mean': 0.0, 'iou_std': 0.0},
            'cuts':   {'p33': 0, 'p66': 0}
        }

    areas_np = np.array(areas)

    p33 = int(np.percentile(areas_np, 33))
    p66 = int(np.percentile(areas_np, 66))

    buckets = {
        'small':  {'dice': [], 'iou': []},
        'medium': {'dice': [], 'iou': []},
        'large':  {'dice': [], 'iou': []},
    }


    for p, g, a in zip(probs_list, gts_list, areas_np):
        pb = (p >= threshold).astype(np.uint8)
        d = _dice_np(pb, g)
        i = _iou_np(pb, g)
        if a <= p33:
            buckets['small']['dice'].append(d)
            buckets['small']['iou'].append(i)
        elif a <= p66:
            buckets['medium']['dice'].append(d)
            buckets['medium']['iou'].append(i)
        else:
            buckets['large']['dice'].append(d)
            buckets['large']['iou'].append(i)


    out = {}
    for k in ['small', 'medium', 'large']:
        d_arr = np.array(buckets[k]['dice']) if buckets[k]['dice'] else np.array([])
        i_arr = np.array(buckets[k]['iou'])  if buckets[k]['iou']  else np.array([])
        out[k] = {
            'count':    int(len(d_arr)),
            'dice_mean': float(d_arr.mean()) if d_arr.size else 0.0,
            'dice_std':  float(d_arr.std(ddof=0)) if d_arr.size else 0.0,
            'iou_mean':  float(i_arr.mean()) if i_arr.size else 0.0,
            'iou_std':   float(i_arr.std(ddof=0)) if i_arr.size else 0.0,
        }

    out['cuts'] = {'p33': int(p33), 'p66': int(p66)}
    return out



def area_ks_test(preds_bin: List[np.ndarray], gts_bin: List[np.ndarray]) -> Dict[str, float]:
    if ks_2samp is None:
        return {"ks_stat": 0.0, "p_value": 1.0}
    pred_areas = np.array([p.sum() for p in preds_bin])
    gt_areas = np.array([g.sum() for g in gts_bin])
    stat, p = ks_2samp(pred_areas, gt_areas)
    return {"ks_stat": float(stat), "p_value": float(p),
            "pred_mean": float(pred_areas.mean()), "gt_mean": float(gt_areas.mean())}
