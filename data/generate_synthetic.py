"""
Synthetic Flood Dataset Generator
==================================
Generates realistic-looking satellite-style flood images + binary masks
using only numpy and opencv — no internet required.

Techniques used:
  - Perlin-like noise via layered sine waves for terrain texture
  - Irregular polygon + flood-fill for water/flood regions
  - Realistic water color (blue-grey tones with shimmer)
  - Vegetation (green patches), soil (brown), urban (grey) background
  - Gaussian blur, brightness variation, and noise for realism
  - Consistent image+mask augmentation for the training split

Usage:
    python data/generate_synthetic.py --output data/flood_dataset --train 400 --val 80 --size 256
"""

import os
import sys
import random
import argparse
import json

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------------

def smooth_noise(h, w, scale=0.05, octaves=4):
    """Generate smooth multi-octave noise in [0, 1]."""
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    frequency = scale
    total_amplitude = 0.0

    for _ in range(octaves):
        # Use random phase offsets per octave for variety
        px = np.random.uniform(0, 100)
        py = np.random.uniform(0, 100)
        xs = np.linspace(px, px + frequency * w, w)
        ys = np.linspace(py, py + frequency * h, h)
        xv, yv = np.meshgrid(xs, ys)
        layer = np.sin(xv) * np.cos(yv) + np.cos(xv * 1.3) * np.sin(yv * 0.7)
        layer = (layer - layer.min()) / (layer.max() - layer.min() + 1e-8)
        noise += amplitude * layer
        total_amplitude += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    return noise / total_amplitude


def gaussian_blob(h, w, cx, cy, sigma_x, sigma_y, angle=0):
    """Create a single Gaussian blob mask."""
    xs = np.arange(w) - cx
    ys = np.arange(h) - cy
    xv, yv = np.meshgrid(xs, ys)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * xv + sin_a * yv
    yr = -sin_a * xv + cos_a * yv
    blob = np.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
    return blob.astype(np.float32)


# ---------------------------------------------------------------------------
# Background terrain
# ---------------------------------------------------------------------------

def make_background(h, w):
    """
    Generate a plausible satellite-style background:
    mix of vegetation (green), soil (brown/tan), and urban (grey).
    """
    terrain = smooth_noise(h, w, scale=0.03, octaves=5)
    detail = smooth_noise(h, w, scale=0.08, octaves=3)
    combined = 0.7 * terrain + 0.3 * detail

    # Base RGB from terrain value
    img = np.zeros((h, w, 3), dtype=np.float32)

    # Vegetation zones (green)
    veg_mask = combined > 0.55
    img[veg_mask] = [
        random.uniform(30, 60),    # R
        random.uniform(80, 130),   # G
        random.uniform(20, 50),    # B
    ]

    # Soil/dry land (tan/brown)
    soil_mask = (combined > 0.35) & ~veg_mask
    img[soil_mask] = [
        random.uniform(120, 180),
        random.uniform(100, 150),
        random.uniform(60, 100),
    ]

    # Low terrain (darker green / wet soil)
    low_mask = ~veg_mask & ~soil_mask
    img[low_mask] = [
        random.uniform(50, 90),
        random.uniform(90, 130),
        random.uniform(50, 80),
    ]

    # Sprinkle some urban patches (grey rectangles)
    num_urban = random.randint(0, 4)
    for _ in range(num_urban):
        ux = random.randint(0, w - 1)
        uy = random.randint(0, h - 1)
        uw = random.randint(10, w // 6)
        uh = random.randint(10, h // 6)
        grey = random.uniform(130, 200)
        img[uy:uy+uh, ux:ux+uw] = [grey, grey * 0.95, grey * 0.9]

    # Add fine texture noise
    texture = np.random.normal(0, 6, (h, w, 3)).astype(np.float32)
    img = np.clip(img + texture, 0, 255)

    # Slight blur for satellite look
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Flood / water region generator
# ---------------------------------------------------------------------------

def make_flood_mask(h, w):
    """
    Generate an irregular flood/water mask using:
    - Random polygon base shapes (rivers, lakes, flood plains)
    - Noise-warped edges for realism
    - Between 1 and 4 overlapping water bodies
    """
    mask = np.zeros((h, w), dtype=np.float32)
    num_bodies = random.randint(1, 4)

    for _ in range(num_bodies):
        body_type = random.choice(["lake", "river", "flood_plain"])

        if body_type == "lake":
            # Elliptical lake
            cx = random.randint(w // 5, 4 * w // 5)
            cy = random.randint(h // 5, 4 * h // 5)
            rx = random.randint(w // 12, w // 4)
            ry = random.randint(h // 12, h // 4)
            angle = random.uniform(0, np.pi)
            body = gaussian_blob(h, w, cx, cy, rx * 0.6, ry * 0.6, angle)
            mask += (body > 0.35).astype(np.float32)

        elif body_type == "river":
            # Winding river as a thick polyline
            river_mask = np.zeros((h, w), dtype=np.uint8)
            # Generate control points
            num_pts = random.randint(4, 8)
            if random.random() > 0.5:
                # Horizontal river
                pts_x = np.linspace(0, w - 1, num_pts).astype(int)
                pts_y = np.random.randint(h // 4, 3 * h // 4, num_pts)
            else:
                # Vertical river
                pts_y = np.linspace(0, h - 1, num_pts).astype(int)
                pts_x = np.random.randint(w // 4, 3 * w // 4, num_pts)

            pts = np.stack([pts_x, pts_y], axis=1).astype(np.int32)
            thickness = random.randint(h // 20, h // 8)
            cv2.polylines(river_mask, [pts], False, 255, thickness=thickness)
            # Blur for smooth edges
            river_mask = cv2.GaussianBlur(river_mask, (15, 15), 0)
            mask += (river_mask > 60).astype(np.float32)

        else:  # flood_plain
            # Large irregular blob
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            rx = random.randint(w // 6, w // 2)
            ry = random.randint(h // 6, h // 2)
            body = gaussian_blob(h, w, cx, cy, rx * 0.5, ry * 0.5)
            # Warp with noise for irregular edges
            noise_warp = smooth_noise(h, w, scale=0.06, octaves=3)
            warped = body * (0.7 + 0.6 * noise_warp)
            mask += (warped > 0.4).astype(np.float32)

    # Add noise to edges for irregular coastlines
    edge_noise = smooth_noise(h, w, scale=0.1, octaves=2)
    mask_noisy = mask + 0.3 * (edge_noise - 0.5)

    binary = (mask_noisy > 0.5).astype(np.uint8)

    # Morphological cleanup: remove tiny specks, smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return binary  # 0 or 1


# ---------------------------------------------------------------------------
# Water appearance
# ---------------------------------------------------------------------------

def apply_water(img, mask):
    """
    Paint water pixels with realistic water colors:
    - Blue-grey base
    - Specular shimmer
    - Slight turbidity variation
    """
    h, w = img.shape[:2]
    water_pixels = mask.astype(bool)

    if not water_pixels.any():
        return img

    # Base water color (blue-grey, varies per scene)
    base_b = random.uniform(100, 180)
    base_g = random.uniform(80, 140)
    base_r = random.uniform(40, 100)

    water_layer = img.copy().astype(np.float32)

    # Apply base color
    water_layer[water_pixels, 0] = base_r
    water_layer[water_pixels, 1] = base_g
    water_layer[water_pixels, 2] = base_b

    # Add shimmer (specular highlights)
    shimmer = smooth_noise(h, w, scale=0.15, octaves=2)
    shimmer_strength = random.uniform(20, 50)
    for c in range(3):
        channel = water_layer[:, :, c]
        channel[water_pixels] += shimmer_strength * shimmer[water_pixels]
        water_layer[:, :, c] = channel

    # Turbidity / sediment variation
    turbidity = smooth_noise(h, w, scale=0.05, octaves=2)
    water_layer[water_pixels, 1] += 20 * turbidity[water_pixels]  # greenish turbidity
    water_layer[water_pixels, 0] += 15 * turbidity[water_pixels]  # brownish turbidity

    water_layer = np.clip(water_layer, 0, 255).astype(np.uint8)

    # Blend original terrain with water (edge blending)
    blend_mask = cv2.GaussianBlur(mask.astype(np.float32) * 255, (7, 7), 0) / 255.0
    blend_mask = blend_mask[:, :, np.newaxis]
    result = (blend_mask * water_layer + (1 - blend_mask) * img.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment(image, mask):
    """Apply consistent augmentation to image+mask pair."""
    # Horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    # Vertical flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    # 90° rotation
    k = random.randint(0, 3)
    if k:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
    # Brightness / contrast
    if random.random() > 0.4:
        alpha = random.uniform(0.8, 1.2)
        beta = random.uniform(-20, 20)
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    # Hue/saturation shift
    if random.random() > 0.5:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    # Gaussian noise
    if random.random() > 0.6:
        noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


# ---------------------------------------------------------------------------
# Single sample generator
# ---------------------------------------------------------------------------

def generate_sample(size):
    """Generate one (image, mask) pair."""
    h = w = size
    bg = make_background(h, w)
    flood_mask = make_flood_mask(h, w)
    image = apply_water(bg, flood_mask)
    # mask: 0=background, 255=flood
    mask_255 = (flood_mask * 255).astype(np.uint8)
    return image, mask_255


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(output_dir, train_count, val_count, size):
    for split, count in [("train", train_count), ("val", val_count)]:
        img_dir = os.path.join(output_dir, split, "images")
        msk_dir = os.path.join(output_dir, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        print(f"Generating {count} {split} samples at {size}x{size}...")

        for i in range(count):
            image, mask = generate_sample(size)

            # Augment training samples (but keep originals too)
            if split == "train" and random.random() > 0.3:
                image, mask = augment(image, mask)

            name = f"synthetic_{split}_{i:04d}.png"
            # Save as BGR for opencv
            cv2.imwrite(os.path.join(img_dir, name),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(msk_dir, name), mask)

            if (i + 1) % 50 == 0 or (i + 1) == count:
                water_pct = (mask > 127).mean() * 100
                print(f"  [{split}] {i+1}/{count} done (last sample water: {water_pct:.1f}%)")

    # Write dataset info
    info = {
        "train_samples": train_count,
        "val_samples": val_count,
        "image_size": size,
        "source": "Synthetic (numpy + opencv)",
        "mask_values": "0=background, 255=flood/water",
    }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset complete:")
    print(f"  Train: {train_count} images  ->  {output_dir}/train/")
    print(f"  Val:   {val_count} images    ->  {output_dir}/val/")
    print(f"\nTrain with:")
    print(f"  py models/train.py --data_dir {output_dir} --epochs 30 --batch_size 4 --input_size {size}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic flood segmentation dataset")
    parser.add_argument("--output", type=str, default="data/flood_dataset",
                        help="Output directory")
    parser.add_argument("--train", type=int, default=400,
                        help="Number of training samples")
    parser.add_argument("--val", type=int, default=80,
                        help="Number of validation samples")
    parser.add_argument("--size", type=int, default=256,
                        help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n=== Synthetic Flood Dataset Generator ===")
    print(f"Output:  {args.output}")
    print(f"Train:   {args.train} samples")
    print(f"Val:     {args.val} samples")
    print(f"Size:    {args.size}x{args.size}")
    print(f"Seed:    {args.seed}\n")

    build_dataset(args.output, args.train, args.val, args.size)
