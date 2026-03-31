"""
Training pipeline for EDGE-DRIVE flood segmentation.

Usage:
    python models/train.py --data_dir data/flood_dataset --epochs 50

Expected data directory structure:
    data/flood_dataset/
        train/
            images/
            masks/
        val/
            images/
            masks/
"""

import os
import sys
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.flood_model import UNetMobileNetV2, DeepLabV3Flood
from models.dataset import FloodDataset
from models.losses import CombinedLoss
from models.evaluate import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg = running_loss / num_batches
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | Loss: {avg:.4f}")

    return running_loss / max(num_batches, 1)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}
    num_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Compute metrics per batch
            metrics = compute_metrics(
                preds.cpu().numpy(),
                masks.cpu().numpy()
            )
            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    avg_loss = running_loss / max(num_batches, 1)
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    return avg_loss, avg_metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Datasets
    train_dataset = FloodDataset(
        image_dir=os.path.join(args.data_dir, "train", "images"),
        mask_dir=os.path.join(args.data_dir, "train", "masks"),
        input_size=args.input_size,
        augment=True,
    )
    val_dataset = FloodDataset(
        image_dir=os.path.join(args.data_dir, "val", "images"),
        mask_dir=os.path.join(args.data_dir, "val", "masks"),
        input_size=args.input_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Model selection
    if args.model == "deeplabv3":
        model = DeepLabV3Flood(num_classes=1, freeze_backbone=args.freeze_backbone).to(device)
        print(f"Using DeepLabV3 (COCO-pretrained, backbone frozen={args.freeze_backbone})")
    else:
        model = UNetMobileNetV2(num_classes=1).to(device)
        print("Using UNetMobileNetV2")

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Loss, optimizer, scheduler
    criterion = CombinedLoss(bce_weight=0.3, dice_weight=0.5, focal_weight=0.2)

    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_iou = 0.0
    history = []

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs-1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **val_metrics,
        })

        # Save history after every epoch (so we can monitor)
        with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  -> Saved best model (IoU: {best_iou:.4f})")

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
            }, save_path)

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EDGE-DRIVE flood segmentation model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--model", type=str, default="deeplabv3", choices=["deeplabv3", "unet"],
                        help="Model architecture (default: deeplabv3)")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone (train only classifier head)")
    args = parser.parse_args()

    train(args)
