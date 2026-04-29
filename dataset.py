"""
dataset.py — IIIT-HW-Dev Devanagari OCR
Updated with stronger data augmentation for better generalization
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from vocab import encode


def get_transforms(split, img_height=32):
    """
    Train: strong augmentation to simulate real handwriting variation
    Val/Test: only resize + normalize
    """
    if split == "train":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_height * 4)),

            # Simulate ink blur / out-of-focus writing
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ], p=0.3),

            # Simulate lighting variation on paper
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4)
            ], p=0.5),

            # Simulate slight pen tilt
            transforms.RandomRotation(degrees=4),

            # Simulate paper warp / scan distortion
            transforms.RandomPerspective(distortion_scale=0.15, p=0.4),

            # Randomly sharpen — simulate dry pen on rough paper
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),

            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_height, img_height * 4)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])


class HindiOCRDataset(Dataset):
    def __init__(self, txt_file, root_dir, img_height=32, split="train"):
        self.root_dir  = root_dir
        self.samples   = []
        self.transform = get_transforms(split, img_height)

        total = 0
        kept  = 0

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue

                img_path, label = parts
                full_path = os.path.join(root_dir, img_path)

                if os.path.exists(full_path):
                    if len(encode(label)) > 0:
                        self.samples.append((img_path, label))
                        kept += 1

        print(f"Total samples: {total}")
        print(f"Usable samples: {kept}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)

        image = Image.open(full_path).convert("L")
        image = self.transform(image)

        label_encoded = encode(label)

        if len(label_encoded) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        label_tensor = torch.tensor(label_encoded, dtype=torch.long)
        return image, label_tensor, len(label_encoded)
