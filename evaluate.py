import argparse
import os
import torch
import torch.nn as nn

from src.data.dataset import get_dataloaders
from src.evaluation.metrics import dice_score, mean_iou, pixel_accuracy
from src.models.model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DoCaSeg model checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--data-root", type=str, default="data/pets_data", help="Path to dataset")
    parser.add_argument("--split", type=str, default="test", choices=["test", "valid"], help="Dataset split to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Train the model first using train.py.")

    print("Loading dataset...")
    train_loader, valid_loader, test_loader, _ = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True
    )
    loader = test_loader if args.split == "test" else valid_loader

    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    print(f"Evaluating checkpoint '{args.checkpoint}' on {args.split} set ({len(loader.dataset)} samples)...")
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / total_samples
    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    acc = pixel_accuracy(cat_preds, cat_targets)
    miou = mean_iou(cat_preds, cat_targets, num_classes=3)
    dice = dice_score(cat_preds, cat_targets, num_classes=3)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Loss:           {avg_loss:.4f}")
    print(f"Pixel Accuracy: {acc * 100:.2f}%")
    print(f"Mean IoU:       {miou * 100:.2f}%")
    print(f"Dice Score:     {dice * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()

