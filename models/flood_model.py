"""
Flood Segmentation Model — Local Training Version
===================================================
U-Net with ResNet50 encoder via segmentation-models-pytorch.
Accepts 3-channel RGB satellite images, outputs a single-channel
flood probability map.

Usage (via train.py):
    py models/train.py --data_dir data/flood_dataset --epochs 30 --batch_size 4 --input_size 256
"""

import os
import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# segmentation-models-pytorch install check
# ---------------------------------------------------------------------------
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    print("Installing segmentation-models-pytorch...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "segmentation-models-pytorch", "-q"]
    )
    import segmentation_models_pytorch as smp

print(f"PyTorch: {torch.__version__}")
print(f"Device : {'GPU — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(device="cpu", encoder="resnet50", in_channels=3, ckpt_path=None):
    """
    Build a U-Net with the specified encoder.

    Args:
        device      : 'cuda' or 'cpu'
        encoder     : encoder backbone name (resnet50, resnet34, efficientnet-b0, etc.)
        in_channels : number of input channels (3 for RGB)
        ckpt_path   : optional path to a saved checkpoint to resume from

    Returns:
        model (nn.Module) moved to `device`
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights="imagenet",   # pretrained ImageNet weights
        in_channels=in_channels,
        classes=1,
        activation=None,              # raw logits — loss / inference applies sigmoid
    )

    model = model.to(device)

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Support both checkpoint formats
        state = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
        if state:
            model.load_state_dict(state)
            epoch = checkpoint.get("epoch", "?")
            iou   = checkpoint.get("best_iou") or checkpoint.get("val_iou", 0.0)
            print(f"Loaded checkpoint: {ckpt_path}  (epoch={epoch}, IoU={iou:.4f})")
        else:
            print(f"Warning: checkpoint at {ckpt_path} has no recognised state dict key.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : smp.Unet / {encoder} encoder")
    print(f"Params  : {n_params:,} trainable")

    return model


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(device=device)
    dummy  = torch.randn(2, 3, 256, 256).to(device)
    out    = model(dummy)
    print(f"\nForward pass OK — input: {dummy.shape}  output: {out.shape}")
