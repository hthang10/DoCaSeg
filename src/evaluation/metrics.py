import torch


def pixel_accuracy(pred, target):
    """
    Calculates pixel accuracy between prediction and target masks.
    Args:
        pred: Tensor of predicted class indices (B, H, W) or (H, W)
        target: Tensor of target class indices (B, H, W) or (H, W)
    Returns:
        float: Pixel accuracy
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total if total > 0 else 0.0


def mean_iou(pred, target, num_classes=3):
    """
    Calculates Mean Intersection over Union (mIoU) across all classes.
    Args:
        pred: Tensor of predicted class indices (B, H, W) or (H, W)
        target: Tensor of target class indices (B, H, W) or (H, W)
        num_classes: int, number of segmentation classes
    Returns:
        float: Mean IoU
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return sum(ious) / len(ious) if len(ious) > 0 else 0.0


def dice_score(pred, target, num_classes=3, smooth=1e-6):
    """
    Calculates Dice Coefficient across all classes.
    """
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()

        if total > 0:
            score = (2.0 * intersection + smooth) / (total + smooth)
            dice_scores.append(score)

    return sum(dice_scores) / len(dice_scores) if len(dice_scores) > 0 else 0.0

