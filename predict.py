import argparse
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.dataset import transform
from src.models.model import UNet
from src.utils.visualization import de_normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using DoCaSeg on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to trained weights")
    parser.add_argument("--output", type=str, default="outputs/predictions/prediction.png", help="Path to save result plot")
    return parser.parse_args()


def predict(image_path, checkpoint_path, output_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load model
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image
    pil_img = Image.open(image_path).convert("RGB")
    tensor_img = transform(pil_img)
    input_tensor = tensor_img.unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    vis_img = de_normalize(tensor_img)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(vis_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="viridis")
    axes[1].set_title("Predicted Segmentation Mask")
    axes[1].axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Prediction saved to {output_path}")

    plt.close(fig)
    return pred_mask


def main():
    args = parse_args()
    predict(args.image, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

