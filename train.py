import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import get_dataloaders
from src.models.model import UNet
from src.training.evaluator import evaluate
from src.utils.seed import set_seed
from src.utils.visualization import display_prediction


def parse_args():
    parser = argparse.ArgumentParser(description="Train DoCaSeg (UNet) on Oxford-IIIT Pet Dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument("--data-root", type=str, default="data/pets_data", help="Directory for dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory to save prediction plots")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    parser.add_argument("--no-vis", action="store_true", help="Disable visual prediction display during training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Thiết lập seed để đảm bảo tính tái lập
    set_seed(args.seed)

    # Tự động chọn thiết bị huấn luyện
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Đảm bảo các thư mục lưu trữ tồn tại
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Tải dữ liệu
    print("Loading Oxford-IIIT Pet dataset...")
    train_loader, valid_loader, test_loader, valid_set = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=True
    )
    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(valid_set)} valid, {len(test_loader.dataset)} test")

    # Khởi tạo mô hình
    model = UNet(n_channels=3, n_classes=3)
    model.to(device)

    # Hàm mất mát và tối ưu hóa
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Chọn một ảnh validation để theo dõi
    display_index = min(80, len(valid_set) - 1)
    display_image, display_target = valid_set[display_index]

    train_losses = []
    valid_losses = []

    best_loss = float("inf")
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming training from checkpoint: {args.resume}")
            checkpoint_wts = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint_wts)
            current_val_loss = evaluate(
                model=model,
                data_loader=valid_loader,
                criterion=criterion,
                device=device,
            )
            best_loss = current_val_loss
            print(f"Loaded checkpoint baseline validation loss: {best_loss:.4f}")
        else:
            print(f"Warning: Checkpoint '{args.resume}' not found. Training from scratch.")

    best_model_wts = copy.deepcopy(model.state_dict())

    max_epoch = args.epochs

    for epoch in range(max_epoch):
        # ===== TRAIN =====
        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples

        # ===== VALIDATION =====
        valid_loss = evaluate(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device,
        )

        # Lưu loss của từng epoch
        train_losses.append(epoch_loss)
        valid_losses.append(valid_loss)

        # Lưu trọng số nếu validation loss tốt hơn
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(
                best_model_wts,
                best_model_path
            )

        print(
            f"Epoch [{epoch + 1}/{max_epoch}] "
            f"Train Loss: {epoch_loss:.4f} "
            f"Valid Loss: {valid_loss:.4f}"
        )

        # Hiển thị dự đoán trên một ảnh validation
        if not args.no_vis:
            print(f"Prediction after epoch {epoch + 1}:")
            save_vis_path = os.path.join(args.output_dir, f"epoch_{epoch + 1}.png")
            display_prediction(
                model,
                display_image,
                display_target,
                device,
                save_path=save_vis_path,
                show=False
            )

    # Khôi phục trọng số tốt nhất sau khi huấn luyện
    model.load_state_dict(best_model_wts)

    print(f"\nBest validation loss: {best_loss:.4f}")

    # Đánh giá cuối cùng trên tập test
    test_loss, test_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        return_metrics=True,
    )

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Pixel Accuracy: {test_metrics['pixel_accuracy'] * 100:.2f}%")
    print(f"Final Test Mean IoU: {test_metrics['mIoU'] * 100:.2f}%")


if __name__ == "__main__":
    main()