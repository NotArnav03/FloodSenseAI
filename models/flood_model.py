import subprocess
subprocess.run(["pip", "install", "segmentation-models-pytorch", "albumentations", "rasterio", "-q"])

import os, warnings, random, zipfile
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import segmentation_models_pytorch as smp
import albumentations as A
from tqdm import tqdm

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")

# ============================================================
# STEP 0 — DOWNLOAD DATA
# ============================================================

dirs = ["S1Hand","LabelHand","S1Weak","LabelWeak"]
for d in dirs:
    os.makedirs(f"/kaggle/working/data/{d}", exist_ok=True)

print("\nDownloading HandLabeled S1...")
os.system("gsutil -m cp gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/*.tif /kaggle/working/data/S1Hand/")
print("Downloading HandLabeled Labels...")
os.system("gsutil -m cp gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/*.tif /kaggle/working/data/LabelHand/")
print("Downloading WeaklyLabeled S1...")
os.system("gsutil -m cp gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1Weak/*.tif /kaggle/working/data/S1Weak/")
print("Downloading WeaklyLabeled Labels...")
os.system("gsutil -m cp gs://sen1floods11/v1.1/data/flood_events/WeaklyLabeled/S1OtsuLabelWeak/*.tif /kaggle/working/data/LabelWeak/")
print("Downloading CSVs...")
os.system("gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_train_data.csv /kaggle/working/data/")
os.system("gsutil cp gs://sen1floods11/v1.1/splits/flood_handlabeled/flood_valid_data.csv /kaggle/working/data/flood_val_data.csv")

s1h  = len(list(Path("/kaggle/working/data/S1Hand").glob("*.tif")))
lh   = len(list(Path("/kaggle/working/data/LabelHand").glob("*.tif")))
s1w  = len(list(Path("/kaggle/working/data/S1Weak").glob("*.tif")))
lw   = len(list(Path("/kaggle/working/data/LabelWeak").glob("*.tif")))
print(f"\nDownload complete: S1Hand={s1h} LabelHand={lh} S1Weak={s1w} LabelWeak={lw}")
assert s1h == 446, f"Expected 446 S1Hand, got {s1h}"
assert s1w == 4384, f"Expected 4384 S1Weak, got {s1w}"

# ============================================================
# STEP 1 — BUILD WEAK CSV
# ============================================================

rows = []
for s1 in sorted(Path("/kaggle/working/data/S1Weak").glob("*.tif")):
    base  = s1.stem.replace("_S1Weak", "")
    label = Path("/kaggle/working/data/LabelWeak") / f"{base}_S1OtsuLabelWeak.tif"
    if label.exists():
        rows.append([s1.name, label.name])
pd.DataFrame(rows).to_csv("/kaggle/working/data/flood_weak_train.csv", index=False, header=False)
print(f"Built weak CSV: {len(rows)} pairs")

# ============================================================
# CONFIG
# ============================================================

CFG = {
    "data_root":      "/kaggle/working/data",
    "train_csv":      "/kaggle/working/data/flood_train_data.csv",
    "val_csv":        "/kaggle/working/data/flood_val_data.csv",
    "weak_csv":       "/kaggle/working/data/flood_weak_train.csv",
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "in_channels":    2,
    "num_workers":    2,
    "weight_decay":   1e-4,
    "seed":           42,
    # Stage 1
    "weak_epochs":    5,
    "weak_lr":        2e-4,
    "weak_batch":     16,
    "weak_save":      "/kaggle/working/pretrained_weak.pth",
    # Stage 2
    "ft_epochs":      80,
    "ft_lr":          1e-4,
    "ft_warmup":      5,
    "ft_batch":       8,
    "ft_save":        "/kaggle/working/best_model.pth",
    # Final combined fine-tune
    "final_epochs":   20,
    "final_lr":       3e-5,
    "final_save":     "/kaggle/working/final_model.pth",
}

SAR_MIN_DB, SAR_MAX_DB = -50.0, 10.0

def sar_normalize(img):
    img = np.nan_to_num(img, nan=-50.0, posinf=10.0, neginf=-50.0)
    img = np.clip(img, SAR_MIN_DB, SAR_MAX_DB)
    return ((img - SAR_MIN_DB) / (SAR_MAX_DB - SAR_MIN_DB)).astype(np.float32)

# ============================================================
# DATASET
# ============================================================

class FloodDataset(Dataset):
    def __init__(self, csv_path, data_root, s1_dir="S1Hand",
                 label_dir="LabelHand", transform=None):
        self.df        = pd.read_csv(csv_path, header=None, names=["s1","label"])
        self.data_root = Path(data_root)
        self.s1_dir    = s1_dir
        self.label_dir = label_dir
        self.transform = transform
        print(f"  Dataset: {len(self.df)} samples | {s1_dir}")

    def __len__(self): return len(self.df)

    def _load_s1(self, fn):
        with rasterio.open(self.data_root / self.s1_dir / Path(fn).name) as src:
            return sar_normalize(src.read().astype(np.float32))

    def _load_label(self, fn):
        with rasterio.open(self.data_root / self.label_dir / Path(fn).name) as src:
            lbl = src.read(1).astype(np.float32)
        return np.clip(np.where(lbl == -1, 0, lbl), 0, 1)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = self._load_s1(row["s1"]).transpose(1, 2, 0)
        mask  = self._load_label(row["label"])
        if self.transform:
            aug  = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        return (torch.from_numpy(img.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask).unsqueeze(0).float())

strong_aug = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.4),
    A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
])
light_aug = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.3),
])

# ============================================================
# LOSS
# ============================================================

class FocalDiceLoss(nn.Module):
    def __init__(self, fw=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.fw, self.gamma, self.alpha = fw, gamma, alpha

    def forward(self, logits, targets):
        logits = torch.clamp(logits, -20.0, 20.0)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        alpha  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal  = (alpha * (1 - p_t) ** self.gamma * bce).mean()
        inter  = (probs * targets).sum(dim=(1,2,3))
        dice   = (1 - (2*inter + 1) / (probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1)).mean()
        loss   = self.fw * focal + (1 - self.fw) * dice
        return loss if not torch.isnan(loss) else torch.tensor(0.5, requires_grad=True, device=logits.device)

def iou(logits, targets, t=0.5):
    p = (torch.sigmoid(logits) > t).float()
    i = (p * targets).sum(dim=(1,2,3))
    u = p.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - i
    return ((i + 1e-6) / (u + 1e-6)).mean().item()

# ============================================================
# TRAIN / VAL
# ============================================================

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tl, ti, n = 0.0, 0.0, 0
    for imgs, masks in tqdm(loader, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, masks)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item(); ti += iou(logits.detach(), masks); n += 1
    return (tl/n, ti/n) if n else (float('nan'), float('nan'))

@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
    model.eval()
    tl, ti = 0.0, 0.0
    for imgs, masks in tqdm(loader, leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        tl += loss_fn(logits, masks).item()
        ti += iou(logits, masks)
    n = len(loader)
    return tl/n, ti/n

def get_scheduler(opt, warmup, total):
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / warmup
        p = (ep - warmup) / max(1, total - warmup)
        return 0.5 * (1 + np.cos(np.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# ============================================================
# MODEL
# ============================================================

def build_model(ckpt_path=None, device="cuda"):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet" if ckpt_path is None else None,
        in_channels=2, classes=1, activation=None,
    ).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded: {ckpt_path} | epoch={ckpt.get('epoch','?')} iou={ckpt.get('val_iou',0):.4f}")
    return model

# ============================================================
# STAGE 1 — WEAK PRETRAIN (5 epochs)
# ============================================================

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
device = CFG["device"]

print(f"\n{'='*55}")
print("STAGE 1 — Weak pretrain (5 epochs, noisy-label safe)")
print(f"{'='*55}")

weak_loader = DataLoader(
    FloodDataset(CFG["weak_csv"], CFG["data_root"], "S1Weak", "LabelWeak", strong_aug),
    batch_size=CFG["weak_batch"], shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)
val_loader = DataLoader(
    FloodDataset(CFG["val_csv"], CFG["data_root"], transform=None),
    batch_size=8, shuffle=False, num_workers=CFG["num_workers"], pin_memory=True)

model   = build_model(device=device)
loss_fn = FocalDiceLoss()
opt     = torch.optim.AdamW(model.parameters(), lr=CFG["weak_lr"], weight_decay=CFG["weight_decay"])
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["weak_epochs"], eta_min=1e-5)

best_s1 = 0.0
for ep in range(1, CFG["weak_epochs"] + 1):
    tl, ti = train_epoch(model, weak_loader, opt, loss_fn, device)
    vl, vi = val_epoch(model, val_loader, loss_fn, device)
    sched.step()
    mark = "  ✓ BEST" if vi > best_s1 else ""
    print(f"[S1] {ep}/{CFG['weak_epochs']} | train loss={tl:.4f} iou={ti:.4f} | val loss={vl:.4f} iou={vi:.4f}{mark}")
    if vi > best_s1:
        best_s1 = vi
        torch.save({"epoch": ep, "model_state": model.state_dict(), "val_iou": vi}, CFG["weak_save"])

print(f"\nStage 1 done. Best val IoU: {best_s1:.4f}")

# ============================================================
# STAGE 2 — FINE-TUNE ON HAND LABELS (80 epochs)
# ============================================================

print(f"\n{'='*55}")
print("STAGE 2 — Fine-tune on hand labels (80 epochs)")
print(f"{'='*55}")

train_loader = DataLoader(
    FloodDataset(CFG["train_csv"], CFG["data_root"], transform=light_aug),
    batch_size=CFG["ft_batch"], shuffle=True, num_workers=CFG["num_workers"], pin_memory=True)

model   = build_model(CFG["weak_save"], device)
loss_fn = FocalDiceLoss()
opt     = torch.optim.AdamW(model.parameters(), lr=CFG["ft_lr"], weight_decay=CFG["weight_decay"])
sched   = get_scheduler(opt, CFG["ft_warmup"], CFG["ft_epochs"])

best_s2 = 0.0
for ep in range(1, CFG["ft_epochs"] + 1):
    tl, ti = train_epoch(model, train_loader, opt, loss_fn, device)
    vl, vi = val_epoch(model, val_loader, loss_fn, device)
    sched.step()
    lr   = opt.param_groups[0]["lr"]
    mark = "  ✓ BEST" if vi > best_s2 else ""
    print(f"[S2] {ep:3d}/{CFG['ft_epochs']} | train loss={tl:.4f} iou={ti:.4f} | val loss={vl:.4f} iou={vi:.4f} | lr={lr:.2e}{mark}")
    if vi > best_s2:
        best_s2 = vi
        torch.save({"epoch": ep, "model_state": model.state_dict(), "val_iou": vi}, CFG["ft_save"])

print(f"\nStage 2 done. Best val IoU: {best_s2:.4f}")

# ============================================================
# THRESHOLD SEARCH
# ============================================================

print(f"\n{'='*55}")
print("Threshold search...")
print(f"{'='*55}")

model = build_model(CFG["ft_save"], device)
model.eval()
val_ds  = FloodDataset(CFG["val_csv"], CFG["data_root"], transform=None)
vloader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

all_probs, all_masks = [], []
with torch.no_grad():
    for imgs, masks in tqdm(vloader, desc="Collecting"):
        all_probs.append(torch.sigmoid(model(imgs.to(device))).squeeze(1).cpu().numpy())
        all_masks.append(masks.squeeze(1).numpy())
all_probs = np.concatenate(all_probs)
all_masks = np.concatenate(all_masks)

best_thresh, best_t_iou = 0.5, 0.0
for t in np.arange(0.10, 0.91, 0.05):
    preds = (all_probs > t).astype(np.float32)
    inter = (preds * all_masks).sum()
    union = preds.sum() + all_masks.sum() - inter
    v     = float((inter + 1e-6) / (union + 1e-6))
    mark  = " <-" if v > best_t_iou else ""
    print(f"  thresh={t:.2f}  IoU={v:.4f}{mark}")
    if v > best_t_iou:
        best_t_iou, best_thresh = v, float(t)

print(f"\nBest threshold: {best_thresh:.2f}  IoU: {best_t_iou:.4f}")

# ============================================================
# FINAL COMBINED FINE-TUNE (train+val, 20 epochs)
# ============================================================

print(f"\n{'='*55}")
print("Final fine-tune on all 341 labeled images (20 epochs)")
print(f"{'='*55}")

all_ds = ConcatDataset([
    FloodDataset(CFG["train_csv"], CFG["data_root"], transform=light_aug),
    FloodDataset(CFG["val_csv"],   CFG["data_root"], transform=light_aug),
])
all_loader = DataLoader(all_ds, batch_size=CFG["ft_batch"], shuffle=True,
                        num_workers=CFG["num_workers"], pin_memory=True)
print(f"  Combined: {len(all_ds)} images")

model   = build_model(CFG["ft_save"], device)
loss_fn = FocalDiceLoss()
opt     = torch.optim.AdamW(model.parameters(), lr=CFG["final_lr"], weight_decay=CFG["weight_decay"])
sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["final_epochs"], eta_min=1e-6)

for ep in range(1, CFG["final_epochs"] + 1):
    model.train()
    tl, n = 0.0, 0
    for imgs, masks in tqdm(all_loader, desc=f"Final {ep:2d}/20", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        opt.zero_grad()
        loss = loss_fn(model(imgs), masks)
        if torch.isnan(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl += loss.item(); n += 1
    sched.step()
    print(f"  Epoch {ep:2d}/20 | loss={tl/max(n,1):.4f} | lr={opt.param_groups[0]['lr']:.2e}")

torch.save({"epoch": "final", "model_state": model.state_dict()}, CFG["final_save"])
print(f"Final model saved: {CFG['final_save']}")

# ============================================================
# TTA INFERENCE
# ============================================================

@torch.no_grad()
def predict_tta(model, tensor):
    model.eval()
    preds = []
    for k in range(4):
        x  = torch.rot90(tensor, k, dims=[2,3])
        p  = torch.sigmoid(model(x.to(device)))
        p  = torch.rot90(p, -k, dims=[2,3])
        preds.append(p.cpu())
        xf = torch.flip(x, dims=[3])
        pf = torch.sigmoid(model(xf.to(device)))
        pf = torch.flip(pf, dims=[3])
        pf = torch.rot90(pf, -k, dims=[2,3])
        preds.append(pf.cpu())
    return torch.stack(preds).mean(0)

# ============================================================
# BATCH INFERENCE — ALL 446 SCENES
# ============================================================

print(f"\n{'='*55}")
print("Batch inference on all 446 scenes...")
print(f"{'='*55}")

final_model = build_model(CFG["final_save"], device)
out_dir     = Path("/kaggle/working/flood_maps")
out_dir.mkdir(exist_ok=True)
tif_files   = sorted(Path("/kaggle/working/data/S1Hand").glob("*.tif"))

failed = []
for tif_path in tqdm(tif_files, desc="Generating maps"):
    try:
        with rasterio.open(tif_path) as src:
            img = sar_normalize(src.read().astype(np.float32))
        tensor   = torch.from_numpy(img).unsqueeze(0).float()
        prob     = predict_tta(final_model, tensor).squeeze().numpy().astype(np.float32)
        out_name = tif_path.stem.replace("_S1Hand", "") + "_flood_prob.npy"
        np.save(out_dir / out_name, prob)
    except Exception as e:
        print(f"  Failed: {tif_path.name} — {e}")
        failed.append(tif_path.name)

npy_files = sorted(out_dir.glob("*.npy"))
print(f"Generated: {len(npy_files)} / {len(tif_files)}")
if failed:
    print(f"Failed: {failed}")

# ============================================================
# ZIP ALL .npy FILES
# ============================================================

zip_path = "/kaggle/working/flood_probability_maps.zip"
print("\nZipping...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in tqdm(npy_files, desc="Zipping"):
        zf.write(f, f.name)
print(f"Zip size: {Path(zip_path).stat().st_size/1024**2:.1f} MB")

# ============================================================
# FINAL SUMMARY
# ============================================================

print(f"\n{'='*55}")
print("FINAL SUMMARY")
print(f"  Stage 1 best val IoU:      {best_s1:.4f}")
print(f"  Stage 2 best val IoU:      {best_s2:.4f}")
print(f"  Post-threshold IoU:        {best_t_iou:.4f}  (thresh={best_thresh:.2f})")
print(f"  Flood maps generated:      {len(npy_files)} / 446")
print(f"  Download: flood_probability_maps.zip")
print(f"{'='*55}")

print("\nSanity check (3 random files):")
for f in random.sample(npy_files, 3):
    arr = np.load(f)
    print(f"  {f.name}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")
