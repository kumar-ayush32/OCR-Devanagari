import torch 
from torch.utils.data import DataLoader
from dataset import HindiOCRDataset

dataset = HindiOCRDataset(
    txt_file="train.txt",
    root_dir="HindiSeg"
)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)          # flat concat, required for CTC
    lengths = torch.tensor(lengths)
    return images, labels, lengths

loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for images, labels, lengths in loader:
    print("Images shape:", images.shape)
    print("Labels:", labels)
    print("Lengths:", lengths)
    break