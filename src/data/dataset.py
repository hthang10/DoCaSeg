import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

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


def get_dataloaders(root="data/pets_data", batch_size=64,
                    num_workers=4):

    train_set = OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    test_set = OxfordIIITPet(
        root=root,
        split="test",
        target_types="segmentation",
        transform=transform,
        target_transform=target_transform,
        download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader