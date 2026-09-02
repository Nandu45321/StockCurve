"""
train.py — Train the Siamese network on data/pairs.npy.

Loss: contrastive loss
  loss = y * d² + (1 - y) * max(0, margin - d)²
  d    = L2 distance between embeddings
  margin = 1.0

Saves best model (lowest val_loss) to data/siamese.pt.
Saves loss curve to data/training_loss.png.

Run: python train.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path so matchers/ is importable
sys.path.insert(0, os.path.dirname(__file__))
from matchers.siamese import SiameseNet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAIRS_PATH  = "data/pairs.npy"
MODEL_PATH  = "data/siamese.pt"
PLOT_PATH   = "data/training_loss.png"
MARGIN      = 1.0
LR          = 1e-3
EPOCHS      = 50
BATCH_SIZE  = 64
VAL_RATIO   = 0.20
RANDOM_SEED = 42

# NOTE: This trains on CPU. Your RX 7600S is AMD — to use it you need
# the ROCm build of PyTorch (pip install torch --index-url https://download.pytorch.org/whl/rocm6.2)
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairsDataset(Dataset):
    """Wraps data/pairs.npy as a PyTorch Dataset."""

    def __init__(self, pairs: list[dict]) -> None:
        """Store the list of pair dicts."""
        self.pairs = pairs

    def __len__(self) -> int:
        """Return number of pairs."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (curve_a, curve_b, label) as tensors."""
        p = self.pairs[idx]
        a = torch.tensor(p["curve_a"], dtype=torch.float32).unsqueeze(0)  # (1, 50)
        b = torch.tensor(p["curve_b"], dtype=torch.float32).unsqueeze(0)  # (1, 50)
        y = torch.tensor(p["label"],   dtype=torch.float32)
        return a, b, y


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

def contrastive_loss(emb_a: torch.Tensor, emb_b: torch.Tensor,
                     labels: torch.Tensor, margin: float = MARGIN) -> torch.Tensor:
    """Contrastive loss: y*d² + (1-y)*max(0, margin - d)²."""
    d = torch.nn.functional.pairwise_distance(emb_a, emb_b)  # (batch,)
    pos = labels       * d.pow(2)
    neg = (1 - labels) * torch.clamp(margin - d, min=0).pow(2)
    return (pos + neg).mean()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def run_epoch(model: SiameseNet, loader: DataLoader,
              optimizer: torch.optim.Optimizer | None) -> float:
    """Run one epoch. optimizer=None means eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    with torch.set_grad_enabled(training):
        for a, b, y in loader:
            a, b, y = a.to(DEVICE), b.to(DEVICE), y.to(DEVICE)
            emb_a, emb_b = model(a, b)
            loss = contrastive_loss(emb_a, emb_b, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load pairs, split, train, save best model and loss plot."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -- Load pairs ----------------------------------------------------------
    print(f"[INFO] Loading {PAIRS_PATH} ...")
    raw   = np.load(PAIRS_PATH, allow_pickle=True)
    pairs = list(raw)
    np.random.shuffle(pairs)
    print(f"[INFO] {len(pairs)} pairs loaded")

    # -- Split ---------------------------------------------------------------
    n_val   = int(len(pairs) * VAL_RATIO)
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    print(f"[INFO] Train: {len(train_p)}  |  Val: {len(val_p)}")

    train_ds = PairsDataset(train_p)
    val_ds   = PairsDataset(val_p)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -- Model + optimizer ---------------------------------------------------
    model     = SiameseNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: {total_params:,} parameters  |  Device: {DEVICE}")
    print()

    # -- Training loop -------------------------------------------------------
    best_val  = float("inf")
    train_hist: list[float] = []
    val_hist:   list[float] = []

    header = f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Best':>6}"
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        tr_loss  = run_epoch(model, train_dl, optimizer)
        val_loss = run_epoch(model, val_dl,   None)

        train_hist.append(tr_loss)
        val_hist.append(val_loss)

        saved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            saved = "*"

        print(f"{epoch:>6}  {tr_loss:>12.6f}  {val_loss:>12.6f}  {saved:>6}")

    print()
    print(f"[INFO] Best val loss : {best_val:.6f}")
    print(f"[INFO] Model saved   : {MODEL_PATH}")

    # -- Loss curve ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, EPOCHS + 1), train_hist, label="Train loss", color="#4C9BE8")
    ax.plot(range(1, EPOCHS + 1), val_hist,   label="Val loss",   color="#FF3D00")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Siamese Network Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)
    print(f"[INFO] Loss plot saved: {PLOT_PATH}")


if __name__ == "__main__":
    main()
