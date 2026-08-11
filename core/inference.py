"""
FloodSense AI — Inference Engine
==================================
Loads a trained U-Net checkpoint and runs flood segmentation
on a single RGB image (file path or numpy array).

Usage:
    from core.inference import InferenceEngine
    engine = InferenceEngine("checkpoints/best_model.pth")
    result = engine.predict(image)   # numpy HxW float32 [0,1]
"""

import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ImageNet normalisation (same as training)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODEL_NOT_FOUND_MSG = (
    "No trained model checkpoint found at '{}'. "
    "Please train the model first:\n"
    "  py models/train.py --data_dir data/flood_dataset --epochs 30 "
    "--batch_size 4 --input_size 256 --workers 0"
)


class InferenceEngine:
    """
    Wraps a trained U-Net model for single-image flood segmentation.

    Args:
        checkpoint_path : path to .pth checkpoint saved by train.py
        input_size      : resize input to this square resolution (match training)
        device          : 'cuda', 'cpu', or None (auto-detect)
        threshold       : sigmoid threshold for binary mask
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        input_size: int = 256,
        device: str = None,
        threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.input_size      = input_size
        self.threshold       = threshold
        self.model           = None
        self.model_meta      = {}

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    # ── model loading ──────────────────────────────────────────────────

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(MODEL_NOT_FOUND_MSG.format(self.checkpoint_path))

        from models.flood_model import build_model

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model = build_model(device=self.device)

        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("model_state")
        )
        if state is None:
            raise KeyError("Checkpoint has no 'model_state_dict' or 'model_state' key.")

        self.model.load_state_dict(state)
        self.model.eval()

        self.model_meta = {
            "epoch":    checkpoint.get("epoch", "?"),
            "best_iou": checkpoint.get("best_iou") or checkpoint.get("val_iou", 0.0),
            "device":   self.device,
        }

    # ── pre / post processing ──────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        image : HxWx3 uint8 RGB numpy array
        returns: 1x3xHxW float32 tensor
        """
        img = cv2.resize(image, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        img = np.transpose(img, (2, 0, 1))           # HWC → CHW
        return torch.from_numpy(img).unsqueeze(0)    # 1CHW

    def _postprocess(self, logits: torch.Tensor) -> np.ndarray:
        """
        logits : 1x1xHxW tensor (raw model output)
        returns: HxW float32 array in [0, 1]  (probability map)
        """
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        return probs.astype(np.float32)

    # ── public API ─────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> dict:
        """
        Run inference on a single RGB image.

        Args:
            image : HxWx3 uint8 numpy array (RGB)

        Returns dict with:
            prob_map   : HxW float32 probability map [0,1]
            binary_mask: HxW uint8 binary mask (0 or 255)
            flood_pct  : percentage of image classified as flood
            confidence : mean probability over flood pixels
        """
        tensor = self._preprocess(image).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        prob_map    = self._postprocess(logits)
        binary_mask = ((prob_map > self.threshold) * 255).astype(np.uint8)
        flood_px    = (binary_mask > 0).sum()
        total_px    = binary_mask.size
        flood_pct   = float(flood_px / total_px * 100)

        flood_probs = prob_map[binary_mask > 0]
        confidence  = float(flood_probs.mean()) if len(flood_probs) > 0 else 0.0

        return {
            "prob_map":    prob_map,
            "binary_mask": binary_mask,
            "flood_pct":   flood_pct,
            "confidence":  confidence,
        }

    def predict_with_overlay(self, image: np.ndarray) -> dict:
        """
        predict() + overlay visualisations.

        Additional keys in return dict:
            overlay        : original image with semi-transparent flood mask
            side_by_side   : [original | mask_heatmap] concatenated HxWx3
        """
        result = self.predict(image)

        # Resize original to match prediction size
        orig_resized = cv2.resize(image, (self.input_size, self.input_size),
                                  interpolation=cv2.INTER_LINEAR)

        # Colour overlay: flood pixels in blue
        overlay      = orig_resized.copy()
        flood_pixels = result["binary_mask"] > 0
        overlay[flood_pixels] = (
            0.4 * overlay[flood_pixels] +
            0.6 * np.array([30, 100, 220], dtype=np.float32)
        ).astype(np.uint8)

        # Heatmap of probability
        prob_u8  = (result["prob_map"] * 255).astype(np.uint8)
        heatmap  = cv2.applyColorMap(prob_u8, cv2.COLORMAP_INFERNO)
        heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        result["overlay"]      = overlay
        result["heatmap"]      = heatmap
        result["orig_resized"] = orig_resized

        return result

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def info(self) -> dict:
        return self.model_meta
