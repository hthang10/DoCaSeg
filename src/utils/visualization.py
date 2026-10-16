import matplotlib.pyplot as plt
import numpy as np
import torch


def de_normalize(
    img,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    """
    De-normalizes an image tensor or numpy array for visualization.
    Accepts:
      - torch.Tensor with shape (C, H, W)
      - numpy.ndarray with shape (H, W, C)
    Returns:
      - numpy.ndarray with shape (H, W, C) in range [0.0, 1.0]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    result = img * std + mean
    result = np.clip(result, 0.0, 1.0)
    return result


def show_image(img, title=None, save_path=None, show=True):
    """Displays or saves a single de-normalized image."""
    fig = plt.figure()
    plt.imshow(de_normalize(img))
    if title:
        plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception:
            plt.show()
    plt.close(fig)


def display_prediction(model, image, target, device, save_path=None, show=True):
    """
    Displays the input image, ground truth mask, and predicted mask side by side.
    Args:
        model: PyTorch segmentation model (e.g. UNet)
        image: Tensor of shape (C, H, W)
        target: Tensor or ndarray of shape (H, W)
        device: torch.device
        save_path: Optional path to save the plot figure
        show: Whether to display the plot interactively (non-blocking)
    """
    model.eval()
    with torch.inference_mode():
        if isinstance(image, torch.Tensor):
            input_tensor = image.unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    img_vis = de_normalize(image)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_vis)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(target, cmap="viridis")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred, cmap="viridis")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        try:
            plt.show(block=False)
            plt.pause(0.5)
        except Exception:
            plt.show()
    plt.close(fig)