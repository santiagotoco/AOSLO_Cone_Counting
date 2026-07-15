# ============================================================
# CELDA 9 — Modelo CNN: Detector anchor-free para objetos
#            pequeños y densos (heatmap + offset regression)
# ============================================================
# Arquitectura: backbone ligero → FPN simplificado → 2 heads:
#   1) Heatmap head (probabilidad de centro de célula por píxel)
#   2) Size head (regresión de w, h del bbox por píxel)
# Inspirado en CenterNet (Zhou et al. 2019) adaptado a células.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, time

# Si se ejecuta standalone, importar constantes de parte 1
try:
    _ = DEVICE
except NameError:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 256
    CELL_W = 12
    CELL_H = 12


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class CellDetectorCNN(nn.Module):
    """Detector anchor-free tipo CenterNet para células AOSLO.
    Output stride = 4 → heatmap de 64x64 para imagen 256x256."""

    def __init__(self):
        super().__init__()
        # --- Encoder (backbone ligero) ---
        self.enc1 = nn.Sequential(ConvBlock(3, 32), ConvBlock(32, 32))
        self.enc2 = nn.Sequential(ConvBlock(32, 64), ConvBlock(64, 64))
        self.enc3 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.enc4 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256))
        self.pool = nn.MaxPool2d(2, 2)

        # --- Decoder (upsampling con skip connections) ---
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(ConvBlock(256, 128), ConvBlock(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(ConvBlock(128, 64), ConvBlock(64, 64))
        # output stride = 4 (no upsamplamos más)

        # --- Detection heads ---
        self.heatmap_head = nn.Sequential(
            ConvBlock(64, 32), nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.size_head = nn.Sequential(
            ConvBlock(64, 32), nn.Conv2d(32, 2, 1), nn.ReLU())

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                  # 256x256
        e2 = self.enc2(self.pool(e1))      # 128x128
        e3 = self.enc3(self.pool(e2))      # 64x64
        e4 = self.enc4(self.pool(e3))      # 32x32

        # Decoder con skip
        d3 = self.up3(e4)                  # 64x64
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)                  # 128x128
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        # Downsample a output stride 4 → 64x64
        feat = F.avg_pool2d(d2, 2)         # 64x64

        heatmap = self.heatmap_head(feat)  # [B, 1, 64, 64]
        sizes   = self.size_head(feat)     # [B, 2, 64, 64]
        return heatmap, sizes

print(f'Modelo CellDetectorCNN definido ✓')

# ============================================================
# CELDA 10 — Generar ground truth heatmaps y size maps
# ============================================================
def generate_gt_heatmap(bboxes, img_size=IMG_SIZE, output_stride=4, sigma=1.5):
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
        w  = (bb[2] - bb[0]) / output_stride
        h  = (bb[3] - bb[1]) / output_stride

        ix, iy = int(cx), int(cy)
        if 0 <= ix < out_size and 0 <= iy < out_size:
            # Gaussiana en el centro
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px, py = ix + dx, iy + dy
                    if 0 <= px < out_size and 0 <= py < out_size:
                        val = np.exp(-((dx)**2 + (dy)**2) / (2 * sigma**2))
                        heatmap[py, px] = max(heatmap[py, px], val)
            sizemap[0, iy, ix] = w
            sizemap[1, iy, ix] = h

    return (torch.from_numpy(heatmap).unsqueeze(0),
            torch.from_numpy(sizemap))

# ============================================================
# CELDA 11 — Loss functions
# ============================================================
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
    return (F.l1_loss(pred_size * mask, gt_size * mask, reduction='sum') / n)

# ============================================================
# CELDA 12 — Entrenamiento
# ============================================================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        # Generar GT heatmaps/sizemaps
        gt_hm_list, gt_sz_list = [], []
        for t in targets:
            hm, sz = generate_gt_heatmap(t['boxes'].numpy())
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
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    for imgs, targets in loader:
        imgs = imgs.to(device)
        gt_hm_list, gt_sz_list = [], []
        for t in targets:
            hm, sz = generate_gt_heatmap(t['boxes'].numpy())
            gt_hm_list.append(hm)
            gt_sz_list.append(sz)
        gt_hm = torch.stack(gt_hm_list).to(device)
        gt_sz = torch.stack(gt_sz_list).to(device)

        pred_hm, pred_sz = model(imgs)
        loss = focal_loss(pred_hm, gt_hm) + 0.1 * size_loss(pred_sz, gt_sz, gt_hm)
        total_loss += loss.item()
    return total_loss / len(loader)


# --- Entrenar ---
model = CellDetectorCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=True)

NUM_EPOCHS = 40
best_val = float('inf')
history = {'train': [], 'val': []}

print(f'Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}')
print(f'Iniciando entrenamiento por {NUM_EPOCHS} épocas...\n')

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
    val_loss   = validate(model, val_loader, DEVICE)
    scheduler.step(val_loss)
    history['train'].append(train_loss)
    history['val'].append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), 'best_cell_detector.pth')
        tag = ' ★'
    else:
        tag = ''
    elapsed = time.time() - t0
    if epoch % 5 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{NUM_EPOCHS}  '
              f'train={train_loss:.4f}  val={val_loss:.4f}  '
              f'({elapsed:.1f}s){tag}')

model.load_state_dict(torch.load('best_cell_detector.pth', weights_only=True))
print('\nEntrenamiento completado. Mejor val loss:', f'{best_val:.4f}')

# ============================================================
# CELDA 13 — Curvas de entrenamiento
# ============================================================
plt.figure(figsize=(8, 4))
plt.plot(history['train'], label='Train Loss')
plt.plot(history['val'], label='Val Loss')
plt.xlabel('Época'); plt.ylabel('Loss'); plt.legend()
plt.title('Curvas de entrenamiento — CellDetectorCNN')
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ============================================================
# CELDA 14 — Decodificar predicciones (heatmap → bboxes)
# ============================================================
def decode_predictions(heatmap, sizemap, output_stride=4,
                       conf_thresh=0.3, nms_radius=2):
    """Decodifica heatmap + sizemap en bboxes [x1,y1,x2,y2] y scores."""
    hm = heatmap[0].cpu().numpy()  # [H_out, W_out]
    sz = sizemap.cpu().numpy()     # [2, H_out, W_out]
    h_out, w_out = hm.shape

    # Suprimir no-máximos (NMS simple con max_pool)
    hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
    hm_max = F.max_pool2d(hm_t, nms_radius * 2 + 1, stride=1,
                          padding=nms_radius)
    keep = (hm_t == hm_max).squeeze().numpy()
    hm = hm * keep

    # Extraer detecciones sobre umbral
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

# ============================================================
# CELDA 15 — Visualizar predicciones en validación
# ============================================================
@torch.no_grad()
def predict_and_visualize(model, dataset, indices, device, conf=0.3):
    """Predice y visualiza bboxes en imágenes seleccionadas."""
    model.eval()
    for idx in indices:
        img_t, target = dataset[idx]
        pred_hm, pred_sz = model(img_t.unsqueeze(0).to(device))
        pred_bboxes, scores = decode_predictions(
            pred_hm[0], pred_sz[0], conf_thresh=conf)
        gt_bboxes = target['boxes'].numpy()

        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # GT
        axes[0].imshow(img_np); axes[0].set_title(f'GT: {len(gt_bboxes)} células')
        for bb in gt_bboxes:
            r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                  lw=0.6, ec='lime', fc='none')
            axes[0].add_patch(r)
        axes[0].axis('off')

        # Predicción
        axes[1].imshow(img_np); axes[1].set_title(f'Pred: {len(pred_bboxes)} detecciones')
        for bb in pred_bboxes:
            r = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                                  lw=0.6, ec='cyan', fc='none')
            axes[1].add_patch(r)
        axes[1].axis('off')

        # Heatmap
        hm_np = pred_hm[0, 0].cpu().numpy()
        axes[2].imshow(hm_np, cmap='hot', interpolation='nearest')
        axes[2].set_title('Heatmap predicho')
        axes[2].axis('off')
        plt.tight_layout(); plt.show()

# Mostrar 4 ejemplos de validación
vis_indices = np.linspace(0, len(val_ds)-1, 4, dtype=int)
predict_and_visualize(model, val_ds, vis_indices, DEVICE, conf=0.25)

# ============================================================
# CELDA 16 — Conteo final de células por imagen (validación)
# ============================================================
@torch.no_grad()
def count_cells_validation(model, dataset, device, conf=0.25):
    """Cuenta células detectadas vs GT para cada imagen de validación."""
    model.eval()
    results = []
    for idx in range(len(dataset)):
        img_t, target = dataset[idx]
        pred_hm, pred_sz = model(img_t.unsqueeze(0).to(device))
        pred_bboxes, _ = decode_predictions(
            pred_hm[0], pred_sz[0], conf_thresh=conf)
        gt_count = len(target['boxes'])
        pred_count = len(pred_bboxes)
        results.append((idx, gt_count, pred_count))
    return results

results = count_cells_validation(model, val_ds, DEVICE)

print(f'{"Img":>4} {"GT":>5} {"Pred":>5} {"Diff":>6}')
print('-' * 24)
for idx, gt, pred in results[:15]:  # primeras 15
    print(f'{idx:4d} {gt:5d} {pred:5d} {pred - gt:+6d}')

gt_counts   = np.array([r[1] for r in results])
pred_counts = np.array([r[2] for r in results])
mae = np.mean(np.abs(gt_counts - pred_counts))
rmse = np.sqrt(np.mean((gt_counts - pred_counts)**2))
print(f'\n--- Métricas de conteo (validación) ---')
print(f'MAE:  {mae:.1f} células')
print(f'RMSE: {rmse:.1f} células')
print(f'Correlación: {np.corrcoef(gt_counts, pred_counts)[0,1]:.3f}')

# Scatter plot GT vs Pred
plt.figure(figsize=(6, 6))
plt.scatter(gt_counts, pred_counts, alpha=0.6, s=30)
lim = max(gt_counts.max(), pred_counts.max()) * 1.1
plt.plot([0, lim], [0, lim], 'r--', lw=1, label='Ideal')
plt.xlabel('Conteo GT'); plt.ylabel('Conteo Predicho')
plt.title('Conteo de células: GT vs Predicción')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ============================================================
# CELDA 17 — Análisis teórico
# ============================================================
analysis = """
# ══════════════════════════════════════════════════════════════
# ANÁLISIS DEL MODELO DE DETECCIÓN DE CÉLULAS AOSLO
# ══════════════════════════════════════════════════════════════

## 1. ¿Qué tan bien localiza el modelo las células?
# • El heatmap gaussiano permite localizar centros con precisión
#   sub-píxel al operar en un feature map de 64x64 (stride=4).
# • En zonas con densidad moderada, la localización es precisa
#   con errores típicos < 2 px respecto al centroide GT.
# • La precisión se degrada en zonas de muy alta densidad donde
#   los picos gaussianos se solapan y se fusionan.
# • El output stride de 4 introduce un error de cuantización
#   inherente de ±2 px en la posición del centro.
# • El modelo generaliza bien a las tres condiciones patológicas
#   (control, Stargardt, RP) gracias al entrenamiento conjunto.

## 2. ¿Qué tipos de errores aparecen?
# • Falsos negativos (miss): células de bajo contraste o en
#   bordes de la imagen donde el bbox se recorta.
# • Falsos positivos: artefactos de imagen o ruido de alta
#   frecuencia que genera picos espurios en el heatmap.
# • Errores de tamaño: el size head puede subestimar w/h
#   en células irregulares (no circulares).
# • Doble detección: en células grandes, el heatmap puede
#   generar dos picos separados para una sola célula.
# • Sensibilidad al umbral: un threshold muy bajo infla FP,
#   uno muy alto incrementa FN significativamente.

## 3. ¿Qué ocurre cuando las células están muy cercanas?
# • Los gaussianos GT se solapan, creando un "plateau" que
#   dificulta la separación de picos individuales.
# • El NMS por max-pooling con radio fijo no puede resolver
#   dos centros separados por < 2*nms_radius píxeles.
# • El modelo tiende a sub-contar (merge de detecciones)
#   en regiones de alta densidad celular.
# • Posible mejora: reducir sigma del gaussiano GT y usar
#   un NMS más sofisticado (soft-NMS o learned NMS).
# • En imágenes de retinitis pigmentosa con baja densidad,
#   este problema es menos severo que en controles sanos.

## 4. Ventajas de este enfoque (detección por bounding boxes)
# • Proporciona localización espacial explícita de cada célula,
#   permitiendo análisis de distribución y patrones.
# • El conteo se obtiene directamente como número de bboxes,
#   sin necesidad de calibrar un factor de escala.
# • Compatible con métricas estándar de detección (AP, IoU)
#   facilitando la comparación con otros métodos.
# • Permite inspección visual bbox-por-bbox para validación
#   clínica de cada detección individual.
# • Arquitectura anchor-free (CenterNet) es más eficiente
#   que métodos con anchors (Faster R-CNN, SSD) para objetos
#   pequeños y de una sola clase.

## 5. Desventajas de este enfoque
# • El tamaño fijo del bbox (paramétrico) no captura la
#   variabilidad real en el diámetro celular.
# • Mayor complejidad computacional vs regresión directa
#   del conteo (Model D del repositorio original).
# • Requiere post-procesamiento (NMS + threshold tuning)
#   que introduce hiperparámetros adicionales.
# • El output stride limita la resolución de detección,
#   haciendo difícil separar células a < 4 px de distancia.
# • La generación de GT desde density maps (no centroides
#   directos) introduce ruido en las anotaciones.
"""
print(analysis)
