"""
Evaluation + Inference for Devanagari OCR
- Runs model on test set
- Plots CER distribution, confusion samples, prediction examples
- Single image inference function
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
from torch.utils.data import DataLoader

from dataset import HindiOCRDataset
from model import CRNN
from vocab import VOCAB_SIZE, ctc_decode, idx_to_char
from train import collate_fn, decode_predictions, decode_targets, cer


# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────

DATASET_ROOT    = r"E:\Project\OCR_Devnagri\ML project"
TEST_TXT        = os.path.join(DATASET_ROOT, "test.txt")
IMAGES_ROOT     = os.path.join(DATASET_ROOT, "HindiSeg")
CHECKPOINT      = os.path.join(DATASET_ROOT, "checkpoints", "checkpoint_epoch25.pth")
RESULTS_DIR     = os.path.join(DATASET_ROOT, "results")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 64
IMG_HEIGHT      = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 2. DEVANAGARI FONT SETUP
# Matplotlib needs a font that supports Unicode Devanagari
# Download Noto Sans Devanagari and place it in the dataset folder,
# OR let the code fall back to showing encoded text
# ─────────────────────────────────────────────

def get_devanagari_font(size=14):
    """Try to load a Devanagari-capable font for matplotlib."""
    candidates = [
        os.path.join(DATASET_ROOT, "NotoSansDevanagari-Regular.ttf"),
        "C:/Windows/Fonts/mangal.ttf",          # built-in Windows Hindi font
        "C:/Windows/Fonts/aparaj.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return fm.FontProperties(fname=path, size=size)
    return fm.FontProperties(size=size)         # fallback — may show boxes


# ─────────────────────────────────────────────
# 3. LOAD MODEL
# ─────────────────────────────────────────────

def load_model(checkpoint_path):
    model = CRNN(vocab_size=VOCAB_SIZE).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_cer = checkpoint.get("val_cer", "?")
    print(f"Loaded checkpoint: epoch {epoch}  |  saved val CER: {val_cer:.4f}")
    return model


# ─────────────────────────────────────────────
# 4. RUN FULL TEST SET EVALUATION
# ─────────────────────────────────────────────

def evaluate_test_set(model, loader):
    """
    Returns:
        all_preds:   list of predicted strings
        all_targets: list of ground truth strings
        all_cers:    list of per-sample CER values
        avg_cer:     float
        avg_wer:     float  (word error rate — 0 if exact match, 1 if wrong)
    """
    all_preds   = []
    all_targets = []
    all_cers    = []

    print("Running evaluation on test set...")

    with torch.no_grad():
        for batch_idx, (images, labels, label_lens) in enumerate(loader):
            images     = images.to(DEVICE)
            log_probs  = model(images)

            preds   = decode_predictions(log_probs)
            targets = decode_targets(labels, label_lens)

            for pred, target in zip(preds, targets):
                all_preds.append(pred)
                all_targets.append(target)
                all_cers.append(cer(pred, target))

            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}")

    avg_cer = np.mean(all_cers)
    avg_wer = np.mean([0.0 if p == t else 1.0
                       for p, t in zip(all_preds, all_targets)])

    print(f"\nTest CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"Test WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Exact matches: {sum(p==t for p,t in zip(all_preds,all_targets))}"
          f" / {len(all_preds)}")

    return all_preds, all_targets, all_cers, avg_cer, avg_wer


# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────

def plot_cer_distribution(all_cers, avg_cer, save_path):
    """Histogram of per-sample CER values."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(all_cers, bins=50, color="#5DCAA5", edgecolor="white", linewidth=0.5)
    ax.axvline(avg_cer, color="#D85A30", linewidth=2,
               label=f"Mean CER = {avg_cer*100:.2f}%")
    ax.axvline(0.0, color="#378ADD", linewidth=1.5, linestyle="--",
               label="Perfect (CER = 0)")

    ax.set_xlabel("Character Error Rate (CER)", fontsize=13)
    ax.set_ylabel("Number of samples", fontsize=13)
    ax.set_title("CER Distribution on Test Set", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotation: percentage of perfect predictions
    perfect_pct = 100 * sum(c == 0 for c in all_cers) / len(all_cers)
    ax.text(0.98, 0.95, f"Perfect predictions: {perfect_pct:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="#3C3489",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEEDFE", alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_cer_buckets(all_cers, save_path):
    """Bar chart showing what % of samples fall in each CER bucket."""
    buckets      = ["Perfect\n(0%)", "Good\n(0-10%)", "Fair\n(10-30%)",
                    "Poor\n(30-60%)", "Wrong\n(>60%)"]
    bucket_colors = ["#1D9E75", "#5DCAA5", "#EF9F27", "#D85A30", "#E24B4A"]
    counts = [
        sum(c == 0.0              for c in all_cers),
        sum(0.0 < c <= 0.1        for c in all_cers),
        sum(0.1 < c <= 0.3        for c in all_cers),
        sum(0.3 < c <= 0.6        for c in all_cers),
        sum(c  > 0.6              for c in all_cers),
    ]
    pcts = [100 * c / len(all_cers) for c in counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(buckets, pcts, color=bucket_colors, edgecolor="white", width=0.6)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=12)

    ax.set_ylabel("Percentage of test samples (%)", fontsize=13)
    ax.set_title("Prediction Quality Breakdown", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(pcts) + 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_prediction_samples(model, dataset, n=12, save_path=None):
    """
    Shows a grid of sample images with predicted vs ground truth labels.
    Green border = correct, Red border = wrong.
    """
    font_prop = get_devanagari_font(size=11)

    # Pick random samples
    indices = np.random.choice(len(dataset), size=n, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    axes = axes.flatten()

    model.eval()
    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            image, label_tensor, label_len = dataset[idx]
            target = "".join([idx_to_char.get(i.item(), "")
                               for i in label_tensor if i.item() != 0])

            # Run inference
            inp = image.unsqueeze(0).to(DEVICE)   # (1, 1, 32, 128)
            log_probs = model(inp)                 # (T, 1, vocab)
            pred = decode_predictions(log_probs)[0]

            # Display image
            img_np = image.squeeze().cpu().numpy()
            img_np = (img_np * 0.5) + 0.5         # undo normalization
            ax.imshow(img_np, cmap="gray")

            is_correct = (pred == target)
            border_color = "#1D9E75" if is_correct else "#E24B4A"

            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"GT:   {target}\nPred: {pred}",
                fontproperties=font_prop,
                color="black",
                loc="left",
                pad=4,
            )

    fig.suptitle("Sample Predictions  |  Green = correct  |  Red = wrong",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_label_length_vs_cer(all_targets, all_cers, save_path):
    """Shows how CER varies with word length."""
    from collections import defaultdict
    length_cers = defaultdict(list)
    for target, c in zip(all_targets, all_cers):
        length_cers[len(target)].append(c)

    lengths  = sorted(length_cers.keys())
    mean_cers = [np.mean(length_cers[l]) for l in lengths]
    counts    = [len(length_cers[l]) for l in lengths]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.plot(lengths, mean_cers, color="#534AB7", linewidth=2.5,
             marker="o", markersize=5, label="Mean CER")
    ax1.fill_between(lengths, mean_cers, alpha=0.15, color="#534AB7")
    ax2.bar(lengths, counts, color="#B5D4F4", alpha=0.5, width=0.8, label="Sample count")

    ax1.set_xlabel("Word length (characters)", fontsize=13)
    ax1.set_ylabel("Mean CER", fontsize=13, color="#534AB7")
    ax2.set_ylabel("Number of samples", fontsize=13, color="#378ADD")
    ax1.set_title("CER vs Word Length", fontsize=15, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#534AB7")
    ax2.tick_params(axis="y", labelcolor="#378ADD")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────
# 6. SINGLE IMAGE INFERENCE
# Use this to test on your own images
# ─────────────────────────────────────────────

def predict_single(model, image_path):
    """
    Run inference on any single image file.
    Usage: predict_single(model, "path/to/word.jpg")
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_HEIGHT, IMG_HEIGHT * 4)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(DEVICE)   # (1, 1, 32, 128)

    model.eval()
    with torch.no_grad():
        log_probs = model(tensor)                        # (T, 1, vocab)
        pred = decode_predictions(log_probs)[0]

    print(f"Image:     {os.path.basename(image_path)}")
    print(f"Predicted: {pred}")
    return pred


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load model
    model = load_model(CHECKPOINT)

    # Test dataset
    test_ds = HindiOCRDataset(TEST_TXT, IMAGES_ROOT, IMG_HEIGHT)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Evaluate
    all_preds, all_targets, all_cers, avg_cer, avg_wer = evaluate_test_set(
        model, test_loader
    )

    # ── Plot 1: CER distribution histogram
    plot_cer_distribution(
        all_cers, avg_cer,
        save_path=os.path.join(RESULTS_DIR, "cer_distribution.png")
    )

    # ── Plot 2: Quality buckets bar chart
    plot_cer_buckets(
        all_cers,
        save_path=os.path.join(RESULTS_DIR, "quality_buckets.png")
    )

    # ── Plot 3: CER vs word length
    plot_label_length_vs_cer(
        all_targets, all_cers,
        save_path=os.path.join(RESULTS_DIR, "cer_vs_length.png")
    )

    # ── Plot 4: Sample predictions grid
    plot_prediction_samples(
        model, test_ds, n=12,
        save_path=os.path.join(RESULTS_DIR, "sample_predictions.png")
    )

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print(f"\nFinal Results:")
    print(f"  Test CER : {avg_cer*100:.2f}%")
    print(f"  Test WER : {avg_wer*100:.2f}%")
    print(f"  Exact match: {sum(p==t for p,t in zip(all_preds,all_targets))}"
          f" / {len(all_preds)}")