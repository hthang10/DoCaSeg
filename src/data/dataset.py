import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

img_size = (128, 128)
num_classes = 3
batch_size = 64

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


def target_transform(target):
    target = transforms.Resize(
        img_size,
        interpolation=transforms.InterpolationMode.NEAREST
    )(target)

    target = transforms.functional.pil_to_tensor(target).squeeze(0)
    target = target.long() - 1

    return target


def get_datasets(root="data/pets_data", val_split=0.2, seed=42, download=True):
    """
    Loads OxfordIIITPet dataset and splits 'trainval' into train_set and valid_set.
    Returns:
        train_set, valid_set, test_set
    """
    trainval_set = OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=download
    )

    test_set = OxfordIIITPet(
        root=root,
        split="test",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=download
    )

    val_size = int(len(trainval_set) * val_split)
    train_size = len(trainval_set) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_set, valid_set = random_split(
        trainval_set,
        [train_size, val_size],
        generator=generator
    )

    return train_set, valid_set, test_set


def get_dataloaders(
    root="data/pets_data",
    batch_size=64,
    val_split=0.2,
    num_workers=2,
    seed=42,
    download=True
):
    """
    Creates DataLoader instances for train, valid, and test sets.
    Returns:
        train_loader, valid_loader, test_loader, valid_set
    """
    train_set, valid_set, test_set = get_datasets(
        root=root,
        val_split=val_split,
        seed=seed,
        download=download
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader, test_loader, valid_set
    return train_loader, valid_loader, test_loader, valid_set