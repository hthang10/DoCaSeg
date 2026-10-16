import pytest
import torch

from src.data.dataset import target_transform, transform
from src.evaluation.metrics import dice_score, mean_iou, pixel_accuracy
from src.models.blocks import ConvBlock, Decoder, Encoder
from src.models.model import UNet


def test_conv_block():
    block = ConvBlock(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 64, 64)
    y = block(x)
    assert y.shape == (2, 64, 64, 64)


def test_encoder():
    encoder = Encoder(in_channels=64, out_channels=128)
    x = torch.randn(2, 64, 64, 64)
    y = encoder(x)
    assert y.shape == (2, 128, 32, 32)


def test_decoder():
    decoder = Decoder(in_channels=128, out_channels=64)
    x1 = torch.randn(2, 128, 16, 16)
    x2 = torch.randn(2, 64, 32, 32)
    y = decoder(x1, x2)
    assert y.shape == (2, 64, 32, 32)


def test_unet_forward():
    model = UNet(n_channels=3, n_classes=3)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 3, 128, 128)


def test_metrics():
    # Perfect match
    pred = torch.tensor([[0, 1], [2, 0]])
    target = torch.tensor([[0, 1], [2, 0]])
    assert pixel_accuracy(pred, target) == 1.0
    assert mean_iou(pred, target, num_classes=3) == 1.0
    assert dice_score(pred, target, num_classes=3) == 1.0

    # Completely wrong match
    pred_wrong = torch.tensor([[1, 2], [0, 1]])
    assert pixel_accuracy(pred_wrong, target) == 0.0
    assert mean_iou(pred_wrong, target, num_classes=3) == 0.0
