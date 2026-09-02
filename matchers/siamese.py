"""
matchers/siamese.py — Siamese network architecture for curve similarity.

Architecture:
  Encoder: three Conv1d layers → AdaptiveAvgPool1d → 64-dim embedding
  SiameseNet: shared encoder, forward returns (emb_a, emb_b)

This module defines the architecture only.
Training lives in train_siamese.py.
The BaseMatcher wrapper lives in matchers/nn.py (stub until trained).
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encodes a single (batch, 1, 50) curve into a (batch, 64) embedding."""

    def __init__(self) -> None:
        """Build three Conv1d blocks followed by global average pooling."""
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1,  out_channels=16, kernel_size=5, padding=2),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Block 3
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Global average pooling → (batch, 64, 1)
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, 1, 50) → returns (batch, 64)."""
        return self.net(x).squeeze(-1)  # (batch, 64)


class SiameseNet(nn.Module):
    """Siamese network with a single shared Encoder for both branches."""

    def __init__(self) -> None:
        """Instantiate one shared Encoder used for both inputs."""
        super().__init__()
        self.encoder = Encoder()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode both curves with shared weights. Returns (emb_a, emb_b)."""
        return self.encoder(a), self.encoder(b)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single curve. x: (batch, 1, 50) → (batch, 64)."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net  = SiameseNet()
    a    = torch.randn(4, 1, 50)
    b    = torch.randn(4, 1, 50)
    ea, eb = net(a, b)
    assert ea.shape == (4, 64), f"Expected (4, 64), got {ea.shape}"
    assert eb.shape == (4, 64), f"Expected (4, 64), got {eb.shape}"
    print("Smoke test passed")
