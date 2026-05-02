"""
HybridDetector: dual-stream (RGB + Event) object detector for drone detection.

Two fusion modes (set via configs/config.yaml):
  early: concatenate RGB + event frame → 4-ch input → shared ResNet18
  late:  separate ResNet18 encoders; feature maps added before FPN

FPN neck → 3-scale YOLO-style detection heads at strides /8, /16, /32.

Output per scale: (B, na, H, W, 5 + num_classes)
  raw logits — apply sigmoid to objectness/class, sigmoid+offset/exp+anchor to boxes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Shared blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

def _make_backbone(in_channels: int, pretrained: bool) -> nn.Module:
    """
    ResNet18 trimmed to return (c2, c3, c4) at strides /8, /16, /32.
    If in_channels != 3 the first conv is replaced; pretrained weights are
    averaged across the input channel dimension to warm-start.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)

    if in_channels != 3:
        old = net.conv1
        new_conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        if pretrained and old.weight is not None:
            with torch.no_grad():
                avg = old.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(avg.expand_as(new_conv.weight))
        net.conv1 = new_conv

    class _BB(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.stem   = nn.Sequential(n.conv1, n.bn1, n.relu, n.maxpool)
            self.layer1 = n.layer1   # stride 4,   64 ch
            self.layer2 = n.layer2   # stride 8,  128 ch
            self.layer3 = n.layer3   # stride 16, 256 ch
            self.layer4 = n.layer4   # stride 32, 512 ch

        def forward(self, x):
            x  = self.stem(x)
            x  = self.layer1(x)
            c2 = self.layer2(x)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)
            return c2, c3, c4

    return _BB(net)


# ---------------------------------------------------------------------------
# FPN neck
# ---------------------------------------------------------------------------

class FPNNeck(nn.Module):
    """
    Top-down FPN fusing layer2 / layer3 / layer4 features.
    Returns (p2, p3, p4) all with `out_ch` channels at strides /8, /16, /32.
    """

    def __init__(self, in_channels: tuple = (128, 256, 512), out_ch: int = 256):
        super().__init__()
        c2, c3, c4 = in_channels
        self.lat4 = ConvBNReLU(c4, out_ch)
        self.lat3 = ConvBNReLU(c3, out_ch)
        self.lat2 = ConvBNReLU(c2, out_ch)
        self.out4 = ConvBNReLU(out_ch, out_ch, 3, p=1)
        self.out3 = ConvBNReLU(out_ch, out_ch, 3, p=1)
        self.out2 = ConvBNReLU(out_ch, out_ch, 3, p=1)

    def forward(self, c2, c3, c4):
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        return self.out2(p2), self.out3(p3), self.out4(p4)


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """Single-scale YOLO-style head → (B, na, H, W, 5+nc)."""

    def __init__(self, in_ch: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.na = num_anchors
        self.nc = num_classes
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, in_ch * 2, 3, p=1),
            nn.Conv2d(in_ch * 2, num_anchors * (5 + num_classes), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        out = self.conv(x)  # (B, na*(5+nc), H, W)
        return (
            out.view(B, self.na, 5 + self.nc, H, W)
               .permute(0, 1, 3, 4, 2)
               .contiguous()
        )  # (B, na, H, W, 5+nc)


# ---------------------------------------------------------------------------
# HybridDetector
# ---------------------------------------------------------------------------

class HybridDetector(nn.Module):
    """
    Input:  (B, 4, H, W)  — channels 0:3 RGB, channel 3 event frame
    Output: list of 3 tensors [(B, na, H/8,  W/8,  5+nc),
                               (B, na, H/16, W/16, 5+nc),
                               (B, na, H/32, W/32, 5+nc)]
    """

    def __init__(
        self,
        num_classes: int = 1,
        fusion_mode: str = "late",
        pretrained: bool = True,
        fpn_ch: int = 256,
        num_anchors: int = 3,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        if fusion_mode == "early":
            self.backbone = _make_backbone(4, pretrained)
        elif fusion_mode == "late":
            self.rgb_backbone   = _make_backbone(3, pretrained)
            self.event_backbone = _make_backbone(1, pretrained)
        else:
            raise ValueError(f"fusion_mode must be 'early' or 'late', got {fusion_mode!r}")

        self.fpn         = FPNNeck((128, 256, 512), fpn_ch)
        self.head_small  = DetectionHead(fpn_ch, num_anchors, num_classes)  # /8
        self.head_medium = DetectionHead(fpn_ch, num_anchors, num_classes)  # /16
        self.head_large  = DetectionHead(fpn_ch, num_anchors, num_classes)  # /32

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        rgb   = x[:, :3]
        event = x[:, 3:]

        if self.fusion_mode == "early":
            c2, c3, c4 = self.backbone(x)
        else:
            c2r, c3r, c4r = self.rgb_backbone(rgb)
            c2e, c3e, c4e = self.event_backbone(event)
            c2, c3, c4 = c2r + c2e, c3r + c3e, c4r + c4e

        p2, p3, p4 = self.fpn(c2, c3, c4)
        return [
            self.head_small(p2),
            self.head_medium(p3),
            self.head_large(p4),
        ]
