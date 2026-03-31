"""
Export trained model to ONNX format for edge deployment.

Supports optional INT8 quantization for further speed gains.

Usage:
    python models/export_onnx.py --model checkpoints/best_model.pth
"""

import os
import sys
import argparse

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.flood_model import UNetMobileNetV2, LightweightFloodSegmentation


def export_onnx(model_path, output_path, input_size=512, use_lightweight=False):
    device = torch.device("cpu")

    if use_lightweight:
        model = LightweightFloodSegmentation()
    else:
        model = UNetMobileNetV2(num_classes=1)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["image"],
        output_names=["flood_mask"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "flood_mask": {0: "batch_size"},
        },
    )

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"ONNX model exported to: {output_path}")
    print(f"ONNX file size: {file_size_mb:.2f} MB")


def quantize_dynamic(onnx_path, output_path):
    """Apply dynamic INT8 quantization via ONNX Runtime (if available)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )
        file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
        print(f"Quantized model saved to: {output_path}")
        print(f"Quantized file size: {file_size_mb:.2f} MB")
    except ImportError:
        print("onnxruntime not installed — skipping quantization")
        print("Install with: pip install onnxruntime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/flood_model.onnx")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--lightweight", action="store_true", help="Export lightweight model")
    parser.add_argument("--quantize", action="store_true", help="Also export INT8 quantized version")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_onnx(args.model, args.output, args.input_size, args.lightweight)

    if args.quantize:
        quant_path = args.output.replace(".onnx", "_int8.onnx")
        quantize_dynamic(args.output, quant_path)
