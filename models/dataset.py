"""
Flood segmentation dataset with heavy augmentation for training robustness.

Supports loading image + mask pairs from a directory structure:
    images/
        img_001.png
        img_002.png
    masks/
        img_001.png
        img_002.png
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, input_size=512, augment=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        # Verify matching masks exist
        self.valid_files = []
        for f in self.image_files:
            mask_path = os.path.join(mask_dir, f)
            # Try common mask naming conventions
            if os.path.exists(mask_path):
                self.valid_files.append(f)
            else:
                base, _ = os.path.splitext(f)
                for ext in ['.png', '.jpg', '.tif']:
                    alt = os.path.join(mask_dir, base + ext)
                    if os.path.exists(alt):
                        self.valid_files.append(f)
                        break

        print(f"Found {len(self.valid_files)} image-mask pairs in {image_dir}")

        # ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.valid_files)

    def _find_mask(self, filename):
        mask_path = os.path.join(self.mask_dir, filename)
        if os.path.exists(mask_path):
            return mask_path
        base, _ = os.path.splitext(filename)
        for ext in ['.png', '.jpg', '.tif']:
            alt = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(alt):
                return alt
        raise FileNotFoundError(f"No mask found for {filename}")

    def _augment(self, image, mask):
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random vertical flip
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-15, 15)     # brightness
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # Random hue/saturation shift (important for water color variation)
        if random.random() > 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random Gaussian noise
        if random.random() > 0.7:
            noise = np.random.normal(0, random.uniform(3, 10), image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random crop and resize (scale augmentation)
        if random.random() > 0.5:
            h, w = image.shape[:2]
            scale = random.uniform(0.7, 1.0)
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            image = image[top:top+new_h, left:left+new_w]
            mask = mask[top:top+new_h, left:left+new_w]

        return image, mask

    def __getitem__(self, idx):
        filename = self.valid_files[idx]

        # Load image
        image_path = os.path.join(self.image_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale, 0 = background, 255 = flood)
        mask_path = self._find_mask(filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Augment
        if self.augment:
            image, mask = self._augment(image, mask)

        # Resize
        image = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Binary mask
        mask = (mask > 127).astype(np.float32)
        mask = mask[np.newaxis, :, :]  # Add channel dim

        return torch.from_numpy(image), torch.from_numpy(mask)
