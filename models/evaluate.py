"""
Evaluation metrics for flood segmentation.

Computes IoU, Dice, Precision, Recall on predicted vs ground truth masks.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """Compute segmentation metrics for a batch of predictions.

    Args:
        preds: numpy array of predicted probabilities or binary masks
        targets: numpy array of ground truth binary masks
        threshold: binarization threshold for predictions
        eps: small constant to avoid division by zero

    Returns:
        dict with iou, dice, precision, recall
    """
    # Binarize
    pred_bin = (preds > threshold).astype(np.float32).flatten()
    target_bin = (targets > threshold).astype(np.float32).flatten()

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


def evaluate_model(model_path, image_dir, mask_dir, input_size=512):
    """Run full evaluation of a trained model on a test set."""
    from models.flood_model import FloodModelRunner

    runner = FloodModelRunner(model_path=model_path, input_size=input_size)

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
    ])

    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": []}

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)

        # Find corresponding mask
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            base, _ = os.path.splitext(filename)
            for ext in ['.png', '.jpg', '.tif']:
                alt = os.path.join(mask_dir, base + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break
            else:
                print(f"  Skipping {filename} — no mask found")
                continue

        # Run inference
        pred_mask = runner.run(image_path, save_vis=False)

        # Load ground truth
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_mask > 127).astype(np.float32)

        # Compute metrics
        metrics = compute_metrics(pred_mask, gt_mask)

        for k in all_metrics:
            all_metrics[k].append(metrics[k])

        print(f"  {filename}: IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}")

    # Summary
    print("\n===== EVALUATION SUMMARY =====")
    for k, v in all_metrics.items():
        if v:
            print(f"  {k.upper():>10}: {np.mean(v):.4f} (± {np.std(v):.4f})")
    print("==============================")

    return {k: float(np.mean(v)) for k, v in all_metrics.items() if v}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate flood segmentation model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--images", type=str, required=True, help="Path to test images")
    parser.add_argument("--masks", type=str, required=True, help="Path to ground truth masks")
    parser.add_argument("--input_size", type=int, default=512)
    args = parser.parse_args()

    evaluate_model(args.model, args.images, args.masks, args.input_size)
