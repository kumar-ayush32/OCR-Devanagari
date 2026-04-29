import os
import torch
from torch.utils.data import DataLoader
from dataset import HindiOCRDataset
from model import CRNN
from vocab import VOCAB_SIZE, ctc_decode, idx_to_char


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

class Config:
    DATASET_ROOT = r"E:\Project\OCR_Devnagri\ML project"
    TEST_TXT     = os.path.join(DATASET_ROOT, "test.txt")
    IMAGES_ROOT  = os.path.join(DATASET_ROOT, "HindiSeg")
    CHECKPOINT   = os.path.join(DATASET_ROOT, "checkpoints", "checkpoint_epoch20.pth")

    BATCH_SIZE = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on: {DEVICE}")


# ─────────────────────────────────────────────
# 2. COLLATE FN (same as training)
# ─────────────────────────────────────────────

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images  = torch.stack(images)
    labels  = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths


# ─────────────────────────────────────────────
# 3. DECODE FUNCTIONS
# ─────────────────────────────────────────────

def decode_predictions(log_probs):
    indices = log_probs.argmax(dim=2)  # (T, B)
    indices = indices.permute(1, 0)    # (B, T)

    preds = []
    for seq in indices:
        preds.append(ctc_decode(seq.tolist()))
    return preds


def decode_targets(labels, lengths):
    targets = []
    offset = 0

    for length in lengths.tolist():
        indices = labels[offset: offset + length].tolist()
        word = "".join([idx_to_char.get(i, "") for i in indices if i != 0])
        targets.append(word)
        offset += length

    return targets


# ─────────────────────────────────────────────
# 4. LOAD MODEL
# ─────────────────────────────────────────────

def load_model(cfg):
    model = CRNN(vocab_size=VOCAB_SIZE).to(cfg.DEVICE)

    checkpoint = torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    print("✅ Model loaded")

    return model


# ─────────────────────────────────────────────
# 5. TEST LOOP
# ─────────────────────────────────────────────

def test():
    cfg = Config()

    dataset = HindiOCRDataset(cfg.TEST_TXT, cfg.IMAGES_ROOT)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

    model = load_model(cfg)

    print("\n🔍 Running inference...\n")

    with torch.no_grad():
        for i, (images, labels, lengths) in enumerate(loader):
            images = images.to(cfg.DEVICE)

            log_probs = model(images)

            preds   = decode_predictions(log_probs)
            targets = decode_targets(labels, lengths)

            # Print a few samples
            for p, t in zip(preds[:5], targets[:5]):
                print(f"GT:   {t}")
                print(f"PRED: {p}")
                print("-" * 30)

            # Only show first batch (remove this later)
            break


# ─────────────────────────────────────────────

if __name__ == "__main__":
    test()