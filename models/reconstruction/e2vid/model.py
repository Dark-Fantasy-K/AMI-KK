"""
E2VID: Events-to-Video recurrent UNet.

Architecture matches rpg_e2vid (Rebecq et al., CVPR 2019 / TPAMI 2021):
  https://github.com/uzh-rpg/rpg_e2vid

Default checkpoint config:
  num_bins=5, base_ch=32, num_encoders=3, skip_type='sum'

State-dict key structure (under 'unet.' prefix in official checkpoint):
  unet.head.conv2d.weight                           [32, 5, 5, 5]
  unet.encoders.{i}.conv.conv2d.weight
  unet.encoders.{i}.recurrent_block.Gates.weight    [4*out, in+out, 3, 3]
  unet.resblocks.{i}.conv1.conv2d.weight            [256, 256, 3, 3]
  unet.resblocks.{i}.conv2.conv2d.weight
  unet.decoders.{i}.conv.conv2d.weight
  unet.pred.conv2d.weight                           [1, 32, 1, 1]
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks (naming mirrors rpg_e2vid so checkpoint keys align)
# ---------------------------------------------------------------------------

class ConvLayer(nn.Module):
    """Conv2d → optional BN → optional ReLU.  Weights stored as self.conv2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM cell.

    Combines input + hidden into a single conv (key: self.Gates) so
    checkpoint weight shape is [4*hidden, in+hidden, k, k].
    """

    def __init__(self, input_size: int, hidden_size: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.Gates = nn.Conv2d(
            input_size + hidden_size,
            4 * hidden_size,
            kernel_size,
            padding=padding,
        )
        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if prev_state is None:
            h = x.new_zeros(B, self.hidden_size, H, W)
            c = x.new_zeros(B, self.hidden_size, H, W)
        else:
            h, c = prev_state

        gates = self.Gates(torch.cat([x, h], dim=1))
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c_new = f_g * c + i_g * g_g
        h_new = o_g * torch.tanh(c_new)
        return h_new, (h_new, c_new)


class RecurrentConvLayer(nn.Module):
    """ConvLayer (strided) → ConvLSTM.  Keys: self.conv, self.recurrent_block."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = ConvLayer(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.recurrent_block = ConvLSTM(out_ch, out_ch, kernel_size=3)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple]:
        x = self.conv(x)
        h, new_state = self.recurrent_block(x, state)
        return h, new_state


class ResidualBlock(nn.Module):
    """Two ConvLayers with residual + ReLU.  Keys: self.conv1, self.conv2."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvLayer(ch, ch, 3, padding=1, activation="relu")
        self.conv2 = ConvLayer(ch, ch, 3, padding=1, activation=None)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv2(self.conv1(x)))


class UpsampleConvLayer(nn.Module):
    """Bilinear ×2 upsample → ConvLayer.  Key: self.conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, padding: int = 0):
        super().__init__()
        self.conv = ConvLayer(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class E2VIDNet(nn.Module):
    """
    Recurrent UNet core (exposed as self.unet in the E2VID wrapper so that
    checkpoint keys like 'unet.head.conv2d.weight' load correctly).

    Forward:
        x  : (B, num_bins, H, W)  voxel grid
        → (B, 1, H, W)  intensity in [0, 1]

    Call reset_states() between independent sequences.
    """

    def __init__(
        self,
        num_bins: int = 5,
        base_ch: int = 32,
        num_encoders: int = 3,
        skip_type: str = "sum",
        num_resblocks: int = 2,
    ):
        super().__init__()
        self.num_encoders = num_encoders
        self.skip_type = skip_type

        # encoder_input_sizes  = [32, 64, 128]
        # encoder_output_sizes = [64, 128, 256]
        enc_in  = [base_ch * (2 ** i)       for i in range(num_encoders)]
        enc_out = [base_ch * (2 ** (i + 1)) for i in range(num_encoders)]

        # Head: full resolution
        self.head = ConvLayer(num_bins, base_ch, kernel_size=5, padding=2)

        # Encoders: stride=2 (no separate MaxPool, matching rpg_e2vid)
        self.encoders = nn.ModuleList([
            RecurrentConvLayer(enc_in[i], enc_out[i], kernel_size=5, stride=2, padding=2)
            for i in range(num_encoders)
        ])

        # Bottleneck residual blocks
        btn = enc_out[-1]   # 256
        self.resblocks = nn.ModuleList([ResidualBlock(btn) for _ in range(num_resblocks)])

        # Decoders: upsample back through encoder scales
        # dec_input_sizes  = [256, 128,  64]
        # dec_output_sizes = [128,  64,  32]
        dec_in  = list(reversed(enc_out))
        dec_out = list(reversed(enc_in))
        self.decoders = nn.ModuleList([
            UpsampleConvLayer(dec_in[i], dec_out[i], kernel_size=5, padding=2)
            for i in range(num_encoders)
        ])

        # Prediction head (no activation — sigmoid applied in forward)
        self.pred = ConvLayer(base_ch, 1, kernel_size=1, activation=None)

        # Recurrent states (reset between sequences)
        self.states: list = [None] * num_encoders

    def reset_states(self) -> None:
        self.states = [None] * self.num_encoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # blocks[0] = head output (H, W, base_ch)
        # blocks[i+1] = encoder i output (H/2^(i+1), W/2^(i+1), enc_out[i])
        blocks = [self.head(x)]
        for i, enc in enumerate(self.encoders):
            feat, self.states[i] = enc(blocks[-1], self.states[i])
            blocks.append(feat)

        # Bottleneck
        z = blocks[-1]
        for rb in self.resblocks:
            z = rb(z)

        # Decode with skip connections from corresponding encoder outputs
        # skip at decoder step i  ← blocks[num_encoders - 1 - i]
        # (enc_{N-1-i} output, which spatially matches decoder output at step i)
        for i, dec in enumerate(self.decoders):
            z = dec(z)
            skip = blocks[self.num_encoders - 1 - i]
            if self.skip_type == "sum":
                z = z + skip
            else:
                z = torch.cat([z, skip], dim=1)

        return torch.sigmoid(self.pred(z))


class E2VID(nn.Module):
    """
    Thin wrapper that adds the 'unet' attribute so that checkpoint keys
    like 'unet.head.conv2d.weight' map directly.
    """

    def __init__(self, num_bins: int = 5, base_ch: int = 32, num_encoders: int = 3):
        super().__init__()
        self.unet = E2VIDNet(num_bins=num_bins, base_ch=base_ch, num_encoders=num_encoders)

    def reset_states(self) -> None:
        self.unet.reset_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_e2vid(weights_path: str | Path, device: torch.device) -> E2VID:
    """
    Load official rpg_e2vid checkpoint into our E2VID model.

    The checkpoint is expected to be a dict with key 'state_dict'
    (standard PyTorch Lightning / rpg_e2vid format).
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # rpg_e2vid stores weights under 'state_dict'
    state_dict = ckpt.get("state_dict", ckpt)

    # Read config if present
    cfg = ckpt.get("config", {}).get("args", {}).get("unet_kwargs", {})
    num_bins    = cfg.get("num_bins", 5)
    base_ch     = cfg.get("base_num_channels", 32)
    num_enc     = cfg.get("num_encoders", 3)

    model = E2VID(num_bins=num_bins, base_ch=base_ch, num_encoders=num_enc)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("E2VID checkpoint missing keys (%d): %s …", len(missing), missing[:3])
    if unexpected:
        log.warning("E2VID checkpoint unexpected keys (%d): %s …", len(unexpected), unexpected[:3])

    model.to(device).eval()
    log.info("E2VID loaded from %s  (bins=%d, base_ch=%d, enc=%d)",
             weights_path, num_bins, base_ch, num_enc)
    return model
