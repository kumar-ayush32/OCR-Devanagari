"""
train.py — Devanagari OCR Training Loop
Updated with:
- 50 epochs
- Smaller batch size (32) for better generalization
- CosineAnnealingLR instead of ReduceLROnPlateau
- split parameter passed to HindiOCRDataset
- cudnn.benchmark for speed
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True   # ~10-20% speedup on fixed input sizes

from dataset import HindiOCRDataset
from model import CRNN
from vocab import VOCAB_SIZE, ctc_decode, idx_to_char


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

class Config:
    DATASET_ROOT = r"E:\Project\OCR_Devnagri\ML project"

    TRAIN_TXT    = os.path.join(DATASET_ROOT, "train.txt")
    VAL_TXT      = os.path.join(DATASET_ROOT, "val.txt")
    IMAGES_ROOT  = os.path.join(DATASET_ROOT, "HindiSeg")
    SAVE_DIR     = os.path.join(DATASET_ROOT, "checkpoints")

    EPOCHS       = 25
    BATCH_SIZE   = 128       # smaller = better generalization on 3050 Ti
    LR           = 8e-4     # slightly lower starting LR
    IMG_HEIGHT   = 32

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# 2. COLLATE FUNCTION
# ─────────────────────────────────────────────

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images  = torch.stack(images)
    labels  = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


# ─────────────────────────────────────────────
# 3. CER METRIC
# ─────────────────────────────────────────────

def cer(pred: str, target: str) -> float:
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dp = list(range(len(pred) + 1))
    for t_char in target:
        new_dp = [dp[0] + 1]
        for j, p_char in enumerate(pred):
            new_dp.append(min(
                dp[j] + (0 if p_char == t_char else 1),
                dp[j + 1] + 1,
                new_dp[-1] + 1,
            ))
        dp = new_dp
    return dp[-1] / len(target)


# ─────────────────────────────────────────────
# 4. DECODE HELPERS
# ─────────────────────────────────────────────

def decode_predictions(log_probs):
    """Greedy decode — (T, B, V) -> list of strings."""
    indices = log_probs.argmax(dim=2).permute(1, 0)   # (B, T)
    return [ctc_decode(seq.tolist()) for seq in indices]


def decode_targets(labels, lengths):
    """Reconstruct label strings from flat labels tensor + lengths."""
    targets = []
    offset  = 0
    for length in lengths.tolist():
        indices = labels[offset: offset + length].tolist()
        word    = "".join([idx_to_char.get(i, "") for i in indices if i != 0])
        targets.append(word)
        offset += length
    return targets


# ─────────────────────────────────────────────
# 5. TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels, label_lens) in enumerate(loader):
        images     = images.to(device)
        labels     = labels.to(device)
        label_lens = label_lens.to(device)

        log_probs  = model(images)                      # (T, B, V)
        T, B       = log_probs.size(0), log_probs.size(1)
        input_lens = torch.full((B,), T, dtype=torch.long).to(device)

        loss = criterion(log_probs, labels, input_lens, label_lens)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}  loss: {loss.item():.4f}")

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 6. VALIDATE ONE EPOCH
# ─────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_cer  = 0.0
    count      = 0

    with torch.no_grad():
        for images, labels, label_lens in loader:
            images     = images.to(device)
            labels_dev = labels.to(device)
            label_lens = label_lens.to(device)

            log_probs  = model(images)
            T, B       = log_probs.size(0), log_probs.size(1)
            input_lens = torch.full((B,), T, dtype=torch.long).to(device)

            loss = criterion(log_probs, labels_dev, input_lens, label_lens)
            total_loss += loss.item()

            preds   = decode_predictions(log_probs)
            targets = decode_targets(labels, label_lens.cpu())

            for pred, target in zip(preds, targets):
                total_cer += cer(pred, target)
                count     += 1

    avg_loss = total_loss / len(loader)
    avg_cer  = total_cer / count if count > 0 else 1.0
    return avg_loss, avg_cer


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    cfg = Config()
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(f"Training on : {cfg.DEVICE}")
    print(f"Vocab size  : {VOCAB_SIZE}")
    print(f"Epochs      : {cfg.EPOCHS}")
    print(f"Batch size  : {cfg.BATCH_SIZE}")
    print(f"LR          : {cfg.LR}")

    # Datasets — pass split so augmentation is applied correctly
    train_ds = HindiOCRDataset(cfg.TRAIN_TXT, cfg.IMAGES_ROOT,
                                cfg.IMG_HEIGHT, split="train")
    val_ds   = HindiOCRDataset(cfg.VAL_TXT,   cfg.IMAGES_ROOT,
                                cfg.IMG_HEIGHT, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=(cfg.DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=(cfg.DEVICE == "cuda"),
    )

    model     = CRNN(vocab_size=VOCAB_SIZE).to(cfg.DEVICE)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

    # CosineAnnealingLR — smoothly decays LR to near-zero over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    best_cer = float("inf")
    print(f"\nStarting training for {cfg.EPOCHS} epochs...\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{cfg.EPOCHS}  (lr={current_lr:.2e})")
        print("─" * 40)

        train_loss          = train_epoch(model, train_loader, optimizer,
                                          criterion, cfg.DEVICE)
        val_loss, val_cer   = validate(model, val_loader, criterion, cfg.DEVICE)

        scheduler.step()

        print(f"\n  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val CER    : {val_cer:.4f}  ({val_cer*100:.1f}%)")

        # Save best model
        if val_cer < best_cer:
            best_cer = val_cer
            path = os.path.join(cfg.SAVE_DIR, "best_model.pth")
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer":            val_cer,
                "val_loss":           val_loss,
            }, path)
            print(f"  Saved best model  (CER: {val_cer:.4f})")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            path = os.path.join(cfg.SAVE_DIR, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_cer":            val_cer,
            }, path)

        print()

    print(f"Training complete.  Best CER: {best_cer*100:.1f}%")


if __name__ == "__main__":
    main()
