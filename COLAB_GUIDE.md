# FloodSense - Google Colab Training Guide

Run the full pipeline (dataset generation, training, evaluation, ONNX export) on Google Colab with free GPU.

---

## Cell 1: Clone the repo and install dependencies

```python
!git clone https://github.com/NotArnav03/FloodSenseAI.git
%cd FloodSenseAI
!git checkout ai-dev
!pip install -q torch torchvision numpy opencv-python-headless psutil onnx onnxruntime albumentations requests
```

---

## Cell 2: Verify GPU is available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

> If this shows CPU, go to **Runtime > Change runtime type > T4 GPU**.

---

## Cell 3: Generate the dataset

Downloads satellite imagery from ESRI + water body masks from OpenStreetMap. Takes ~10-15 min.

```python
!python data/generate_dataset.py --output data/flood_dataset --samples 300 --aug 3 --size 512
```

This creates:
- `data/flood_dataset/train/images/` and `masks/` (~1000+ augmented samples)
- `data/flood_dataset/val/images/` and `masks/` (~24 samples)

---

## Cell 4: Train - Phase 1 (frozen backbone)

Fine-tune only the classifier head first. Fast convergence with COCO-pretrained features.

```python
!python models/train.py \
    --data_dir data/flood_dataset \
    --model deeplabv3 \
    --freeze_backbone \
    --epochs 15 \
    --batch_size 16 \
    --lr 1e-3 \
    --workers 2 \
    --output_dir checkpoints
```

> With a T4 GPU, this should take ~5-10 min.

---

## Cell 5: Train - Phase 2 (unfreeze backbone)

Resume from Phase 1 checkpoint and fine-tune the entire network with a lower learning rate.

```python
!python models/train.py \
    --data_dir data/flood_dataset \
    --model deeplabv3 \
    --resume checkpoints/best_model.pth \
    --epochs 30 \
    --batch_size 8 \
    --lr 1e-4 \
    --workers 2 \
    --output_dir checkpoints
```

> Lower batch size (8) because full backprop through backbone uses more VRAM.
> This should take ~15-25 min on T4.

---

## Cell 6: Evaluate the trained model

```python
!python models/evaluate.py \
    --model checkpoints/best_model.pth \
    --images data/flood_dataset/val/images \
    --masks data/flood_dataset/val/masks \
    --input_size 512
```

---

## Cell 7: Run inference on a sample image

```python
import os
from models.flood_model import FloodModelRunner

# Pick the first val image as a sample
val_dir = "data/flood_dataset/val/images"
sample_image = os.path.join(val_dir, os.listdir(val_dir)[0])
print(f"Running inference on: {sample_image}")

runner = FloodModelRunner(
    model_path="checkpoints/best_model.pth",
    model_type="deeplabv3",
    input_size=512
)
mask = runner.run(sample_image, save_vis=True)
```

---

## Cell 8: Visualize the result

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np

overlay = cv2.imread("data/processed/flood_overlay.png")
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

original = cv2.imread(sample_image)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(original)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(overlay)
axes[1].set_title("Flood Detection Overlay")
axes[1].axis("off")

plt.tight_layout()
plt.show()
```

---

## Cell 9: Export to ONNX

```python
!python models/export_onnx.py \
    --model checkpoints/best_model.pth \
    --output checkpoints/flood_model.onnx \
    --model_type deeplabv3 \
    --input_size 512 \
    --quantize
```

This creates:
- `checkpoints/flood_model.onnx` - full precision model
- `checkpoints/flood_model_int8.onnx` - quantized for edge deployment

---

## Cell 10: Benchmark inference speed

```python
!python benchmarks/benchmark_cpu.py
```

---

## Cell 11: Download the trained model

```python
from google.colab import files

files.download("checkpoints/best_model.pth")
files.download("checkpoints/flood_model.onnx")
files.download("checkpoints/training_history.json")
files.download("data/processed/flood_overlay.png")
```

---

## Cell 12 (Optional): View training curves

```python
import json
import matplotlib.pyplot as plt

with open("checkpoints/training_history.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_loss = [h["train_loss"] for h in history]
val_loss = [h["val_loss"] for h in history]
iou = [h["iou"] for h in history]
dice = [h["dice"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss, label="Train Loss")
ax1.plot(epochs, val_loss, label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, iou, label="IoU")
ax2.plot(epochs, dice, label="Dice")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_title("Segmentation Metrics")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

---

## Quick Reference

| Step | Cell | Time (T4 GPU) |
|------|------|---------------|
| Setup | 1-2 | ~1 min |
| Dataset | 3 | ~10-15 min |
| Train Phase 1 | 4 | ~5-10 min |
| Train Phase 2 | 5 | ~15-25 min |
| Evaluate + Visualize | 6-8 | ~1 min |
| ONNX Export | 9 | ~1 min |
| Download | 11 | ~1 min |
| **Total** | | **~35-55 min** |
