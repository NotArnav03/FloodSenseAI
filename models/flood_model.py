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

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.encoder = backbone.features

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
        return image, tensor, original_h, original_w

    def cnn_inference(self, tensor):
        with torch.no_grad():
            output = self.model(tensor)
        return output.squeeze().detach().numpy()

    def water_color_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([140, 255, 255])

        lower_brown = np.array([5, 50, 50])
        upper_brown = np.array([25, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        combined = cv2.bitwise_or(blue_mask, brown_mask)

        combined = combined.astype(np.float32) / 255.0
        return combined

    def edge_enhancement(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        return edges

    def fuse_outputs(self, cnn_mask, color_mask, edge_mask, original_h, original_w):
        cnn_resized = cv2.resize(cnn_mask, (original_w, original_h))

        fused = (
            0.4 * cnn_resized +
            0.4 * color_mask +
            0.2 * edge_mask
        )

        fused = cv2.GaussianBlur(fused, (7, 7), 0)
        fused = np.clip(fused, 0.0, 1.0)
        return fused.astype(np.float32)

    def save_output(self, mask):
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/flood_mask.npy", mask)

    def run(self, image_path):
        image, tensor, h, w = self.preprocess(image_path)

        cnn_mask = self.cnn_inference(tensor)
        color_mask = self.water_color_mask(image)
        edge_mask = self.edge_enhancement(image)

        final_mask = self.fuse_outputs(cnn_mask, color_mask, edge_mask, h, w)

        self.save_output(final_mask)

        print("Flood mask saved.")
        print("Shape:", final_mask.shape)
        print("dtype:", final_mask.dtype)
        print("Min:", final_mask.min(), "Max:", final_mask.max())


if __name__ == "__main__":
    runner = FloodModelRunner()
    runner.run("data/raw/sample_flood.png")