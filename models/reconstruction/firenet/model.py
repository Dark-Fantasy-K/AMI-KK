"""
FireNet: Fast recurrent event-to-frame reconstruction.

Reference:
  Scheerlinck et al., "Fast Image Reconstruction with an Event Camera", WACV 2020
  Checkpoint from: https://github.com/cedric-scheerlinck/rpg_e2vid (firenet branch)

Architecture: head → G1 → G2 → G3 → pred  (all at full resolution, no downsampling)

Expected checkpoint keys (no module prefix in official firenet checkpoint):
  head.conv.weight  / head.conv.bias        [base_ch, num_bins, 3, 3]
  G1.ih.weight / G1.ih.bias                 [4*base_ch, base_ch, 3, 3]
  G1.hh.weight / G1.hh.bias                 [4*base_ch, base_ch, 3, 3]
  G2.*, G3.*
  pred.conv.weight / pred.conv.bias          [1, base_ch, 1, 1]

Note: some checkpoints use 'Gates' (single combined conv) instead of separate
ih/hh. The load_firenet() function tries both naming schemes.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvReLU(nn.Module):
    """Conv2d → BN → ReLU. Stored as self.conv (for checkpoint compat.)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=True)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell with separate input-hidden (ih) and
    hidden-hidden (hh) convolutions — matches firenet checkpoint format.

    Falls back to combined 'Gates' conv if that's what the checkpoint has.
    """

    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        p = k // 2
        # Separate ih / hh convolutions (official firenet format)
        self.ih = nn.Conv2d(ch, 4 * ch, k, padding=p, bias=True)
        self.hh = nn.Conv2d(ch, 4 * ch, k, padding=p, bias=False)
        self.ch = ch

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, _, H, W = x.shape
        if state is None:
            h = x.new_zeros(B, self.ch, H, W)
            c = x.new_zeros(B, self.ch, H, W)
        else:
            h, c = state

        gates = self.ih(x) + self.hh(h)
        i_g, f_g, g_g, o_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        g_g = torch.tanh(g_g)
        o_g = torch.sigmoid(o_g)
        c_new = f_g * c + i_g * g_g
        h_new = o_g * torch.tanh(c_new)
        return h_new, (h_new, c_new)


# ---------------------------------------------------------------------------
# FireNet
# ---------------------------------------------------------------------------

class FireNet(nn.Module):
    """
    FireNet reconstruction network.

    Forward:
        x  : (B, num_bins, H, W)  voxel grid
        → (B, 1, H, W)  intensity in [0, 1]

    Call reset_states() between independent sequences.
    """

    def __init__(self, num_bins: int = 5, base_ch: int = 16):
        super().__init__()
        self.head = ConvReLU(num_bins, base_ch)
        self.G1   = ConvLSTMCell(base_ch)
        self.G2   = ConvLSTMCell(base_ch)
        self.G3   = ConvLSTMCell(base_ch)
        self.pred = nn.Conv2d(base_ch, 1, kernel_size=1, bias=True)

        self.states: list = [None, None, None]

    def reset_states(self) -> None:
        self.states = [None, None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x, self.states[0] = self.G1(x, self.states[0])
        x, self.states[1] = self.G2(x, self.states[1])
        x, self.states[2] = self.G3(x, self.states[2])
        return torch.sigmoid(self.pred(x))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_firenet(weights_path: str | Path, device: torch.device) -> FireNet:
    """
    Load official FireNet checkpoint.

    Handles two checkpoint key formats:
      A) Separate ih/hh convolutions  (standard firenet)
      B) Combined 'Gates' conv         (some variants)
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Read config if present
    cfg      = ckpt.get("config", {}).get("args", {}) if isinstance(ckpt, dict) else {}
    num_bins = cfg.get("num_bins", 5)
    base_ch  = cfg.get("base_num_channels", 16)

    model = FireNet(num_bins=num_bins, base_ch=base_ch)

    # --- Format B: combined Gates → remap to ih / hh ---
    # Some checkpoints store a single conv [4*ch, 2*ch, k, k] under G1.Gates.*
    remapped = {}
    for k, v in state_dict.items():
        if ".Gates." in k:
            # Split combined [4*ch, 2*ch, k, k] into ih [4*ch, ch, k, k]
            # and hh [4*ch, ch, k, k]
            ch_half = v.shape[1] // 2
            prefix = k.replace(".Gates.", ".ih.")
            remapped[prefix] = v[:, :ch_half]
            remapped[k.replace(".Gates.", ".hh.")] = v[:, ch_half:]
        else:
            remapped[k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        log.warning("FireNet checkpoint missing keys (%d): %s …", len(missing), missing[:3])
    if unexpected:
        log.warning("FireNet checkpoint unexpected keys (%d): %s …", len(unexpected), unexpected[:3])

    model.to(device).eval()
    log.info("FireNet loaded from %s", weights_path)
    return model
