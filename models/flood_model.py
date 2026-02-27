import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2


class LightweightFloodSegmentation(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained MobileNetV2 backbone
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.encoder = backbone.features

        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )

        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        out = self.upsample(out)
        out = self.sigmoid(out)
        return out


class FloodModelRunner:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = LightweightFloodSegmentation().to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        tensor = self.transform(image).unsqueeze(0)
        return tensor, original_h, original_w

    def postprocess(self, output, original_h, original_w):
        mask = output.squeeze().detach().numpy()
        mask = cv2.resize(mask, (original_w, original_h))
        mask = np.clip(mask, 0.0, 1.0)
        return mask.astype(np.float32)

    def save_output(self, mask):
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/flood_mask.npy", mask)

    def run(self, image_path):
        tensor, h, w = self.preprocess(image_path)

        with torch.no_grad():
            output = self.model(tensor)

        mask = self.postprocess(output, h, w)
        self.save_output(mask)

        print("Flood mask saved.")
        print("Shape:", mask.shape)
        print("dtype:", mask.dtype)
        print("Min:", mask.min(), "Max:", mask.max())


if __name__ == "__main__":
    runner = FloodModelRunner()
    runner.run("data/raw/sample_flood.jpg")