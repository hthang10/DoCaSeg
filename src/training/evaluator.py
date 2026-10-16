import torch
from src.evaluation.metrics import mean_iou, pixel_accuracy


@torch.inference_mode()
def evaluate(model, data_loader, criterion, device, return_metrics=False, num_classes=3):
    """
    Evaluates the model on a given data loader.
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: torch.device
        return_metrics: bool, if True returns (loss, metrics_dict), else returns loss float
        num_classes: int, number of segmentation classes
    Returns:
        float or tuple: loss or (loss, {"pixel_acc": ..., "mIoU": ...})
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        batch_size = inputs.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if return_metrics:
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(labels.cpu())

    if total_samples == 0:
        raise ValueError("DataLoader không có dữ liệu để đánh giá.")

    avg_loss = total_loss / total_samples

    if return_metrics:
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        acc = pixel_accuracy(cat_preds, cat_targets)
        miou = mean_iou(cat_preds, cat_targets, num_classes=num_classes)
        return avg_loss, {"pixel_accuracy": acc, "mIoU": miou}

    return avg_loss

    if return_metrics:
        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        acc = pixel_accuracy(cat_preds, cat_targets)
        miou = mean_iou(cat_preds, cat_targets, num_classes=num_classes)
        return avg_loss, {"pixel_accuracy": acc, "mIoU": miou}

    return avg_loss

