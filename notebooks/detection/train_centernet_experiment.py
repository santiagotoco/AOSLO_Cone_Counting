#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_centernet_experiment.py
─────────────────────────────
Diseño experimental: 20 iteraciones del modelo CenterNet/Heatmap
para detección de células AOSLO con bounding boxes paramétricos 12×12 px.

Requisitos (Taller 7):
  • 20 iteraciones con semillas 0-19
  • 25 épocas por iteración
  • Métricas: Precision, Recall, mAP@50, Conteo MAE
  • Exporta results/centernet_runs.csv y results/centernet_stats.csv
  • Genera figuras comparativas y boxplots en figures/
"""

import os
import sys
import time
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin GUI para no bloquear
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.ndimage import label, maximum_filter, generate_binary_structure
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 0. RUTAS Y CONSTANTES
# ══════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)           # Celulas Boxes/
REPO_DIR = os.path.join(PROJECT_DIR, "AOSLO-Cell-Density-Estimation")
DATA_DIR = os.path.join(REPO_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "Training+Density")
VAL_DIR = os.path.join(DATA_DIR, "Validation+Density")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros del modelo / bounding box
IMG_SIZE = 256
CELL_W = 12
CELL_H = 12
OUTPUT_STRIDE = 4

# Diseño experimental
NUM_RUNS = 20
NUM_EPOCHS = 25
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
CONF_THRESH = 0.25
IOU_THRESH = 0.50  # para mAP@50

print(f"Device: {DEVICE}")
print(f"Train dir: {TRAIN_DIR} — existe: {os.path.isdir(TRAIN_DIR)}")
print(f"Val   dir: {VAL_DIR}   — existe: {os.path.isdir(VAL_DIR)}")


# ══════════════════════════════════════════════════════════════
# 1. FUNCIONES DE DATOS  (extraídas de nb_part1_data.py)
# ══════════════════════════════════════════════════════════════
def extract_centroids_from_density(density_path, threshold_ratio=0.15):
    """Extrae coordenadas (x, y) de centroides desde un mapa de densidad."""
    density = np.array(Image.open(density_path)).astype(np.float32)
    if density.ndim == 3:
        density = density[:, :, 0]
    if density.max() == 0:
        return np.array([]).reshape(0, 2)
    thresh = density.max() * threshold_ratio
    struct = generate_binary_structure(2, 2)
    local_max = maximum_filter(density, footprint=struct) == density
    detected = (density > thresh) & local_max
    labeled, n_feat = label(detected)
    centroids = []
    for i in range(1, n_feat + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = ys.mean(), xs.mean()
        centroids.append([cx, cy])
    return np.array(centroids) if centroids else np.array([]).reshape(0, 2)


def centroids_to_bboxes(centroids, img_h=IMG_SIZE, img_w=IMG_SIZE,
                        cell_w=CELL_W, cell_h=CELL_H):
    """Convierte centroides (x, y) en bboxes [x_min, y_min, x_max, y_max]."""
    if len(centroids) == 0:
        return np.array([]).reshape(0, 4)
    hw, hh = cell_w / 2.0, cell_h / 2.0
    bboxes = []
    for cx, cy in centroids:
        x_min = max(0, cx - hw)
        y_min = max(0, cy - hh)
        x_max = min(img_w - 1, cx + hw)
        y_max = min(img_h - 1, cy + hh)
        bboxes.append([x_min, y_min, x_max, y_max])
    return np.array(bboxes, dtype=np.float32)


def list_image_density_pairs(directory):
    """Retorna lista de tuplas (img_path, density_path) ordenadas."""
    img_files = sorted(glob(os.path.join(directory, "*.tif")))
    pairs = []
    for f in img_files:
        basename = os.path.basename(f)
        if "Density" not in basename:
            base = os.path.splitext(f)[0]
            density_f = base + "Density.tif"
            if not os.path.exists(density_f):
                density_f = base + "_Density.tif"
            if os.path.exists(density_f):
                pairs.append((f, density_f))
    return pairs


def build_target(bboxes):
    """Construye target dict compatible con detección."""
    if len(bboxes) == 0:
        return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64)}
    return {
        "boxes": torch.tensor(bboxes, dtype=torch.float32),
        "labels": torch.ones(len(bboxes), dtype=torch.int64),
    }


class AOSLODetectionDataset(Dataset):
    """Dataset que retorna (imagen_tensor, target_dict) para detección."""

    def __init__(self, pairs, img_size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, density_path = self.pairs[idx]
        img = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        centroids = extract_centroids_from_density(density_path)
        bboxes = centroids_to_bboxes(centroids, img.shape[0], img.shape[1])

        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1, :].copy()
            if len(bboxes) > 0:
                w = img.shape[1]
                x_min_new = w - 1 - bboxes[:, 2]
                x_max_new = w - 1 - bboxes[:, 0]
                bboxes[:, 0] = x_min_new
                bboxes[:, 2] = x_max_new

        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float()
        target = build_target(bboxes)
        return img_t, target


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return imgs, targets


# ══════════════════════════════════════════════════════════════
# 2. MODELO CenterNet  (extraído de nb_part2_model.py)
# ══════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CellDetectorCNN(nn.Module):
    """Detector anchor-free tipo CenterNet para células AOSLO.
    Output stride = 4 → heatmap de 64×64 para imagen 256×256."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(3, 32), ConvBlock(32, 32))
        self.enc2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64))
        self.enc3 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.enc4 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256))
        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(256, 128), ConvBlock(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(128, 64), ConvBlock(64, 64))

        self.heatmap_head = nn.Sequential(
            ConvBlock(64, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
        self.size_head = nn.Sequential(
            ConvBlock(64, 32), nn.Conv2d(32, 2, 1), nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        feat = F.avg_pool2d(d2, 2)
        heatmap = self.heatmap_head(feat)
        sizes = self.size_head(feat)
        return heatmap, sizes


# ══════════════════════════════════════════════════════════════
# 3. GT GENERATION & LOSS  (extraído de nb_part2_model.py)
# ══════════════════════════════════════════════════════════════
def generate_gt_heatmap(bboxes, img_size=IMG_SIZE, output_stride=OUTPUT_STRIDE,
                        sigma=1.5):
    """Genera heatmap GT gaussiano y size map desde bboxes."""
    out_size = img_size // output_stride
    heatmap = np.zeros((out_size, out_size), dtype=np.float32)
    sizemap = np.zeros((2, out_size, out_size), dtype=np.float32)
    if len(bboxes) == 0:
        return (torch.from_numpy(heatmap).unsqueeze(0),
                torch.from_numpy(sizemap))
    for bb in bboxes:
        cx = (bb[0] + bb[2]) / 2.0 / output_stride
        cy = (bb[1] + bb[3]) / 2.0 / output_stride
        w = (bb[2] - bb[0]) / output_stride
        h = (bb[3] - bb[1]) / output_stride
        ix, iy = int(cx), int(cy)
        if 0 <= ix < out_size and 0 <= iy < out_size:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < out_size and 0 <= py < out_size:
                        val = np.exp(-((dx) ** 2 + (dy) ** 2) / (2 * sigma ** 2))
                        heatmap[py, px] = max(heatmap[py, px], val)
            sizemap[0, iy, ix] = w
            sizemap[1, iy, ix] = h
    return (torch.from_numpy(heatmap).unsqueeze(0),
            torch.from_numpy(sizemap))


def focal_loss(pred, gt, alpha=2, beta=4):
    """Focal loss para heatmap (CornerNet-style)."""
    pos_mask = gt.eq(1).float()
    neg_mask = gt.lt(1).float()
    pred = pred.clamp(1e-6, 1 - 1e-6)
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = (-torch.log(1 - pred) * torch.pow(pred, alpha)
                * torch.pow(1 - gt, beta) * neg_mask)
    n_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos


def size_loss(pred_size, gt_size, gt_heatmap):
    """L1 loss en posiciones con objeto (centro)."""
    mask = (gt_heatmap > 0.99).float().expand_as(pred_size)
    n = mask.sum().clamp(min=1)
    return F.l1_loss(pred_size * mask, gt_size * mask, reduction="sum") / n


# ══════════════════════════════════════════════════════════════
# 4. ENTRENAMIENTO / VALIDACIÓN
# ══════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        gt_hm_list, gt_sz_list = [], []
        for t in targets:
            hm, sz = generate_gt_heatmap(t["boxes"].numpy())
            gt_hm_list.append(hm)
            gt_sz_list.append(sz)
        gt_hm = torch.stack(gt_hm_list).to(device)
        gt_sz = torch.stack(gt_sz_list).to(device)
        pred_hm, pred_sz = model(imgs)
        loss_hm = focal_loss(pred_hm, gt_hm)
        loss_sz = size_loss(pred_sz, gt_sz, gt_hm)
        loss = loss_hm + 0.1 * loss_sz
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate_loss(model, loader, device):
    model.eval()
    total_loss = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        gt_hm_list, gt_sz_list = [], []
        for t in targets:
            hm, sz = generate_gt_heatmap(t["boxes"].numpy())
            gt_hm_list.append(hm)
            gt_sz_list.append(sz)
        gt_hm = torch.stack(gt_hm_list).to(device)
        gt_sz = torch.stack(gt_sz_list).to(device)
        pred_hm, pred_sz = model(imgs)
        loss = focal_loss(pred_hm, gt_hm) + 0.1 * size_loss(pred_sz, gt_sz, gt_hm)
        total_loss += loss.item()
    return total_loss / len(loader)


# ══════════════════════════════════════════════════════════════
# 5. DECODIFICACIÓN  (heatmap → bboxes)
# ══════════════════════════════════════════════════════════════
def decode_predictions(heatmap, sizemap, output_stride=OUTPUT_STRIDE,
                       conf_thresh=CONF_THRESH, nms_radius=2):
    """Decodifica heatmap + sizemap en bboxes [x1,y1,x2,y2] y scores."""
    hm = heatmap[0].cpu().numpy()
    sz = sizemap.cpu().numpy()
    hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    hm_max = F.max_pool2d(hm_t, nms_radius * 2 + 1, stride=1,
                          padding=nms_radius)
    keep = (hm_t == hm_max).squeeze().numpy()
    hm = hm * keep
    ys, xs = np.where(hm > conf_thresh)
    scores = hm[ys, xs]
    bboxes = []
    for x, y in zip(xs, ys):
        w = max(sz[0, y, x], 1.0) * output_stride
        h = max(sz[1, y, x], 1.0) * output_stride
        cx = x * output_stride + output_stride / 2
        cy = y * output_stride + output_stride / 2
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(IMG_SIZE - 1, cx + w / 2)
        y2 = min(IMG_SIZE - 1, cy + h / 2)
        bboxes.append([x1, y1, x2, y2])
    bboxes = np.array(bboxes).reshape(-1, 4) if bboxes else np.zeros((0, 4))
    return bboxes, scores


# ══════════════════════════════════════════════════════════════
# 6. MÉTRICAS: Precision, Recall, mAP@50, Conteo MAE
# ══════════════════════════════════════════════════════════════
def compute_iou(box_a, box_b):
    """IoU entre dos bboxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred_bboxes, pred_scores, gt_bboxes, iou_thresh=IOU_THRESH):
    """Asigna predicciones a GT. Retorna (TP, FP, FN)."""
    if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
        return 0, 0, 0
    if len(pred_bboxes) == 0:
        return 0, 0, len(gt_bboxes)
    if len(gt_bboxes) == 0:
        return 0, len(pred_bboxes), 0

    # Ordenar predicciones por score descendente
    order = np.argsort(-pred_scores)
    pred_bboxes = pred_bboxes[order]

    matched_gt = set()
    tp = 0
    fp = 0
    for pb in pred_bboxes:
        best_iou = 0
        best_j = -1
        for j, gb in enumerate(gt_bboxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_bboxes) - len(matched_gt)
    return tp, fp, fn


def compute_ap_single_image(pred_bboxes, pred_scores, gt_bboxes,
                            iou_thresh=IOU_THRESH):
    """AP para una sola imagen con interpolación all-points."""
    if len(gt_bboxes) == 0:
        return 1.0 if len(pred_bboxes) == 0 else 0.0
    if len(pred_bboxes) == 0:
        return 0.0

    order = np.argsort(-pred_scores)
    pred_bboxes = pred_bboxes[order]
    pred_scores_sorted = pred_scores[order]

    matched_gt = set()
    tps = []
    fps = []
    for pb in pred_bboxes:
        best_iou = 0
        best_j = -1
        for j, gb in enumerate(gt_bboxes):
            if j in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tps.append(1)
            fps.append(0)
            matched_gt.add(best_j)
        else:
            tps.append(0)
            fps.append(1)

    tps_cum = np.cumsum(tps)
    fps_cum = np.cumsum(fps)
    recalls = tps_cum / len(gt_bboxes)
    precisions = tps_cum / (tps_cum + fps_cum)

    # Interpolación all-points (VOC2010+)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


@torch.no_grad()
def evaluate_full(model, dataset, device, conf_thresh=CONF_THRESH):
    """Evalúa el modelo en todo el dataset.
    Retorna dict con Precision, Recall, mAP@50, Count_MAE."""
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    aps = []
    count_errors = []

    for idx in range(len(dataset)):
        img_t, target = dataset[idx]
        pred_hm, pred_sz = model(img_t.unsqueeze(0).to(device))
        pred_bboxes, pred_scores = decode_predictions(
            pred_hm[0], pred_sz[0], conf_thresh=conf_thresh
        )
        gt_bboxes = target["boxes"].numpy()

        # TP / FP / FN
        tp, fp, fn = match_predictions(pred_bboxes, pred_scores, gt_bboxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # AP por imagen
        ap = compute_ap_single_image(pred_bboxes, pred_scores, gt_bboxes)
        aps.append(ap)

        # Error de conteo
        count_errors.append(abs(len(pred_bboxes) - len(gt_bboxes)))

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    mAP50 = np.mean(aps) if aps else 0.0
    count_mae = np.mean(count_errors) if count_errors else 0.0

    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "mAP@50": round(mAP50, 4),
        "Count_MAE": round(count_mae, 2),
    }


# ══════════════════════════════════════════════════════════════
# 7. FIGURAS COMPARATIVAS (predicción vs GT)
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def save_comparison_figure(model, dataset, device, run_id,
                           n_examples=4, conf_thresh=CONF_THRESH):
    """Guarda una figura comparativa GT vs Pred vs Heatmap."""
    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_examples, dtype=int)

    fig, axes = plt.subplots(n_examples, 3, figsize=(18, 5 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img_t, target = dataset[idx]
        pred_hm, pred_sz = model(img_t.unsqueeze(0).to(device))
        pred_bboxes, scores = decode_predictions(
            pred_hm[0], pred_sz[0], conf_thresh=conf_thresh
        )
        gt_bboxes = target["boxes"].numpy()
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # GT
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"GT: {len(gt_bboxes)} células", fontsize=10)
        for bb in gt_bboxes:
            r = patches.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                lw=0.6, ec="lime", fc="none"
            )
            axes[row, 0].add_patch(r)
        axes[row, 0].axis("off")

        # Predicción
        axes[row, 1].imshow(img_np)
        axes[row, 1].set_title(f"Pred: {len(pred_bboxes)} detecciones", fontsize=10)
        for bb in pred_bboxes:
            r = patches.Rectangle(
                (bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1],
                lw=0.6, ec="cyan", fc="none"
            )
            axes[row, 1].add_patch(r)
        axes[row, 1].axis("off")

        # Heatmap
        hm_np = pred_hm[0, 0].cpu().numpy()
        axes[row, 2].imshow(hm_np, cmap="hot", interpolation="nearest")
        axes[row, 2].set_title("Heatmap predicho", fontsize=10)
        axes[row, 2].axis("off")

    fig.suptitle(f"CenterNet — Run {run_id} (seed={run_id})", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"centernet_comparison_run{run_id:02d}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Figura comparativa guardada: {path}")


# ══════════════════════════════════════════════════════════════
# 8. SEMILLAS Y REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    """Fija semilla en torch, numpy y random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════
# 9. BUCLE EXPERIMENTAL PRINCIPAL
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  EXPERIMENTO CENTERNET — 20 ITERACIONES × 25 ÉPOCAS")
    print("=" * 60)

    # Cargar pares de datos
    train_pairs = list_image_density_pairs(TRAIN_DIR)
    val_pairs = list_image_density_pairs(VAL_DIR)
    print(f"\nPares train: {len(train_pairs)},  val: {len(val_pairs)}\n")

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        print("ERROR: No se encontraron pares imagen/densidad.")
        print("Asegúrate de que el repositorio AOSLO-Cell-Density-Estimation "
              "está clonado con los datos.")
        sys.exit(1)

    all_results = []

    for run in range(NUM_RUNS):
        seed = run
        set_seed(seed)
        print(f"\n{'─' * 55}")
        print(f"  RUN {run + 1}/{NUM_RUNS}  |  seed = {seed}")
        print(f"{'─' * 55}")

        # Recrear datasets (para que el augment use la nueva semilla)
        train_ds = AOSLODetectionDataset(train_pairs, augment=True)
        val_ds = AOSLODetectionDataset(val_pairs, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=0)

        # Instanciar modelo fresco
        model = CellDetectorCNN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        best_state = None

        t_start = time.time()
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
            val_loss = validate_loss(model, val_loader, DEVICE)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:3d}/{NUM_EPOCHS}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}")

        elapsed = time.time() - t_start
        print(f"  Entrenamiento completado en {elapsed:.1f}s  "
              f"(mejor val_loss={best_val_loss:.4f})")

        # Cargar mejores pesos
        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluar
        metrics = evaluate_full(model, val_ds, DEVICE)
        metrics["run"] = run
        metrics["seed"] = seed
        metrics["best_val_loss"] = round(best_val_loss, 4)
        all_results.append(metrics)

        print(f"  ► Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  "
              f"mAP@50={metrics['mAP@50']:.4f}  "
              f"Count_MAE={metrics['Count_MAE']:.2f}")

        # Guardar figura comparativa (solo para runs 0, 9, 19)
        if run in [0, 9, 19]:
            save_comparison_figure(model, val_ds, DEVICE, run_id=run)

    # ──────────────────────────────────────────────────────────
    # 10. EXPORTAR RESULTADOS
    # ──────────────────────────────────────────────────────────
    df_runs = pd.DataFrame(all_results)
    cols_order = ["run", "seed", "Precision", "Recall", "mAP@50",
                  "Count_MAE", "best_val_loss"]
    df_runs = df_runs[cols_order]

    runs_csv = os.path.join(RESULTS_DIR, "centernet_runs.csv")
    df_runs.to_csv(runs_csv, index=False)
    print(f"\n✓ Resultados por iteración guardados en: {runs_csv}")

    # Estadísticas resumen
    metric_cols = ["Precision", "Recall", "mAP@50", "Count_MAE"]
    stats_rows = []
    for col in metric_cols:
        stats_rows.append({
            "metric": col,
            "mean": round(df_runs[col].mean(), 4),
            "std": round(df_runs[col].std(), 4),
            "min": round(df_runs[col].min(), 4),
            "max": round(df_runs[col].max(), 4),
        })
    df_stats = pd.DataFrame(stats_rows)
    stats_csv = os.path.join(RESULTS_DIR, "centernet_stats.csv")
    df_stats.to_csv(stats_csv, index=False)
    print(f"✓ Resumen estadístico guardado en: {stats_csv}")

    # Imprimir resumen
    print("\n" + "=" * 55)
    print("  RESUMEN FINAL — CenterNet (20 runs × 25 epochs)")
    print("=" * 55)
    for _, row in df_stats.iterrows():
        print(f"  {row['metric']:>12s}: {row['mean']:.4f} ± {row['std']:.4f}  "
              f"[{row['min']:.4f} – {row['max']:.4f}]")

    # ──────────────────────────────────────────────────────────
    # 11. BOXPLOTS DE DISTRIBUCIÓN ESTADÍSTICA
    # ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
    titles = ["Precision", "Recall", "mAP@50", "Count MAE"]

    for i, (col, title, color) in enumerate(zip(metric_cols, titles, colors)):
        bp = axes[i].boxplot(df_runs[col].values, patch_artist=True,
                             widths=0.5,
                             boxprops=dict(facecolor=color, alpha=0.6),
                             medianprops=dict(color="black", linewidth=2),
                             flierprops=dict(marker="o", markersize=5))
        # Superponer puntos individuales
        x_jitter = np.random.normal(1, 0.04, size=len(df_runs))
        axes[i].scatter(x_jitter, df_runs[col].values, c=color,
                        edgecolors="black", s=30, zorder=5, alpha=0.8)
        mean_val = df_runs[col].mean()
        std_val = df_runs[col].std()
        axes[i].set_title(f"{title}\n{mean_val:.4f} ± {std_val:.4f}",
                          fontsize=12, fontweight="bold")
        axes[i].set_ylabel(title)
        axes[i].set_xticks([1])
        axes[i].set_xticklabels(["CenterNet"])
        axes[i].grid(axis="y", alpha=0.3)

    fig.suptitle("Distribución de métricas — CenterNet (20 runs)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    boxplot_path = os.path.join(FIGURES_DIR, "centernet_boxplots.png")
    fig.savefig(boxplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✓ Boxplots guardados en: {boxplot_path}")

    # ──────────────────────────────────────────────────────────
    # 12. GRÁFICA DE EVOLUCIÓN POR RUN
    # ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (col, title, color) in enumerate(zip(metric_cols, titles, colors)):
        ax = axes[i // 2, i % 2]
        ax.plot(df_runs["run"], df_runs[col], "o-", color=color,
                linewidth=1.5, markersize=5)
        ax.axhline(y=df_runs[col].mean(), color=color, linestyle="--",
                   alpha=0.5, label=f"Media = {df_runs[col].mean():.4f}")
        ax.fill_between(
            df_runs["run"],
            df_runs[col].mean() - df_runs[col].std(),
            df_runs[col].mean() + df_runs[col].std(),
            alpha=0.15, color=color, label=f"±1σ"
        )
        ax.set_xlabel("Run (seed)")
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(NUM_RUNS))

    fig.suptitle("Métricas por iteración — CenterNet", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    evolution_path = os.path.join(FIGURES_DIR, "centernet_metrics_evolution.png")
    fig.savefig(evolution_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Gráfica de evolución guardada en: {evolution_path}")

    print("\n✅ Experimento finalizado exitosamente.\n")


if __name__ == "__main__":
    main()
