# """
# CRNN Model for Devanagari OCR
# Architecture: CNN -> BiLSTM -> CTC
# """

# import torch
# import torch.nn as nn
# from vocab import VOCAB_SIZE


# # ─────────────────────────────────────────────
# # 1. CNN BACKBONE
# # Extracts visual features from the image.
# # Input:  (B, 1, 32, 128)   — batch, channels, height, width
# # Output: (B, 512, 1, 32)   — 512 feature maps, height collapsed to 1
# # ─────────────────────────────────────────────

# class CNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.features = nn.Sequential(
#             # Block 1 — (B, 1,   32, 128) -> (B, 64,  16, 64)
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 2 — (B, 64,  16, 64)  -> (B, 128, 8,  32)
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),

#             # Block 3 — (B, 128, 8,  32)  -> (B, 256, 4,  32)
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # only shrink height

#             # Block 4 — (B, 256, 4,  32)  -> (B, 512, 1,  32)
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # collapse height to 1
#         )

#     def forward(self, x):
#         return self.features(x)   # (B, 512, 1, 32)


# # ─────────────────────────────────────────────
# # 2. BiLSTM
# # Reads the CNN feature sequence left-to-right AND right-to-left.
# # Input:  (T, B, input_size)   — time steps, batch, features
# # Output: (T, B, hidden*2)     — both directions concatenated
# # ─────────────────────────────────────────────

# class BiLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
#         super().__init__()

#         self.lstm = nn.LSTM(
#             input_size  = input_size,
#             hidden_size = hidden_size,
#             num_layers  = num_layers,
#             bidirectional = True,       # forward + backward
#             batch_first = False,        # expects (T, B, features)
#             dropout = dropout if num_layers > 1 else 0,
#         )

#     def forward(self, x):
#         out, _ = self.lstm(x)    # (T, B, hidden*2)
#         return out


# # ─────────────────────────────────────────────
# # 3. FULL CRNN MODEL
# # Wires CNN -> reshape -> BiLSTM -> Linear
# # ─────────────────────────────────────────────

# class CRNN(nn.Module):
#     def __init__(
#         self,
#         vocab_size   = VOCAB_SIZE,   # 97 (96 chars + blank at index 0)
#         lstm_hidden  = 256,          # hidden units per direction
#         lstm_layers  = 2,            # stacked LSTM layers
#         dropout      = 0.3,
#     ):
#         super().__init__()

#         self.cnn = CNN()

#         # CNN outputs 512 channels — that becomes the LSTM input size
#         self.bilstm = BiLSTM(
#             input_size  = 512,
#             hidden_size = lstm_hidden,
#             num_layers  = lstm_layers,
#             dropout     = dropout,
#         )

#         # Project BiLSTM output (hidden*2 = 512) to vocab size
#         self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

#         self._init_weights()

#     def _init_weights(self):
#         for name, param in self.named_parameters():
#             if "weight_ih" in name:
#                 nn.init.xavier_uniform_(param)
#             elif "weight_hh" in name:
#                 nn.init.orthogonal_(param)
#             elif "bias" in name:
#                 nn.init.zeros_(param)

#     def forward(self, x):
#         # x: (B, 1, 32, 128)

#         # --- CNN ---
#         features = self.cnn(x)          # (B, 512, 1, 32)

#         # --- Reshape for LSTM ---
#         B, C, H, W = features.size()
#         assert H == 1, f"CNN height should be 1 after pooling, got {H}"

#         features = features.squeeze(2)  # (B, 512, 32)  — drop height dim
#         features = features.permute(2, 0, 1)  # (32, B, 512) — time x batch x feat

#         # --- BiLSTM ---
#         lstm_out = self.bilstm(features)   # (32, B, 512)

#         # --- Classifier ---
#         logits = self.classifier(lstm_out) # (32, B, 97)

#         # CTCLoss expects log-softmax probabilities
#         log_probs = logits.log_softmax(dim=2)  # (32, B, 97)

#         return log_probs   # (T, B, vocab_size)


# # ─────────────────────────────────────────────
# # 4. QUICK SANITY CHECK
# # Run this file directly to verify shapes are correct
# # ─────────────────────────────────────────────

# if __name__ == "__main__":
#     model = CRNN()
#     print(model)
#     print()

#     # Count parameters
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters:     {total:,}")
#     print(f"Trainable parameters: {trainable:,}")
#     print()

#     # Dummy forward pass
#     batch_size = 4
#     dummy_input = torch.randn(batch_size, 1, 32, 128)

#     with torch.no_grad():
#         output = model(dummy_input)

#     print(f"Input shape:  {dummy_input.shape}")   # (4, 1, 32, 128)
#     print(f"Output shape: {output.shape}")        # (32, 4, 97)
#     print()
#     print("Output shape breakdown:")
#     print(f"  {output.shape[0]} = time steps (= image width after CNN)")
#     print(f"  {output.shape[1]} = batch size")
#     print(f"  {output.shape[2]} = vocab size (log-softmax over characters)")
#     print()
#     print("Model is ready for training.")

"""
model.py — Devanagari OCR CRNN
Updated with deeper CNN (5 blocks) + attention between CNN and BiLSTM
Architecture: CNN -> Attention -> BiLSTM -> Linear -> CTC
"""

import torch
import torch.nn as nn
from vocab import VOCAB_SIZE


# ─────────────────────────────────────────────
# 1. CNN BACKBONE (5 blocks — deeper than before)
# Input:  (B, 1,   32, 128)
# Output: (B, 512,  1,  32)
# ─────────────────────────────────────────────

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — (B, 1,   32, 128) -> (B, 64,  16, 64)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 — (B, 64,  16, 64)  -> (B, 128, 8,  32)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 — (B, 128, 8,  32)  -> (B, 256, 4,  32)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 4 — (B, 256, 4,  32)  -> (B, 512, 2,  32)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Block 5 — (B, 512, 2,  32)  -> (B, 512, 1,  32)
            # Extra conv without pooling to deepen without shrinking width
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, x):
        return self.features(x)   # (B, 512, 1, 32)


# ─────────────────────────────────────────────
# 2. COLUMN ATTENTION
# Learns which time-steps (columns) to focus on.
# Input:  (T, B, H)
# Output: (T, B, H)  — same shape, but re-weighted
# ─────────────────────────────────────────────

class ColumnAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (T, B, H)
        weights = torch.softmax(self.attn(x), dim=0)   # (T, B, 1)
        return x * weights                              # (T, B, H)


# ─────────────────────────────────────────────
# 3. BiLSTM
# Input:  (T, B, input_size)
# Output: (T, B, hidden*2)
# ─────────────────────────────────────────────

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = False,
            dropout       = dropout if num_layers > 1 else 0,
        )

    def forward(self, x):
        out, _ = self.lstm(x)   # (T, B, hidden*2)
        return out


# ─────────────────────────────────────────────
# 4. FULL CRNN MODEL
# CNN -> Attention -> BiLSTM -> Linear
# ─────────────────────────────────────────────

class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size  = VOCAB_SIZE,
        lstm_hidden = 256,
        lstm_layers = 2,
        dropout     = 0.3,
    ):
        super().__init__()

        self.cnn       = CNN()
        self.attention = ColumnAttention(hidden_size=512)
        self.bilstm    = BiLSTM(
            input_size  = 512,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            dropout     = dropout,
        )
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (B, 1, 32, 128)

        # --- CNN ---
        features = self.cnn(x)              # (B, 512, 1, 32)

        B, C, H, W = features.size()
        assert H == 1, f"Expected H=1 after CNN, got H={H}"

        features = features.squeeze(2)      # (B, 512, 32)
        features = features.permute(2, 0, 1)  # (32, B, 512)  time x batch x feat

        # --- Attention ---
        features = self.attention(features) # (32, B, 512)

        # --- BiLSTM ---
        lstm_out = self.bilstm(features)    # (32, B, 512)
        lstm_out = self.dropout(lstm_out)

        # --- Classifier ---
        logits    = self.classifier(lstm_out)          # (32, B, vocab_size)
        log_probs = logits.log_softmax(dim=2)          # (32, B, vocab_size)

        return log_probs


# ─────────────────────────────────────────────
# 5. SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = CRNN()
    print(model)
    print()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()

    dummy = torch.randn(4, 1, 32, 128)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"  {out.shape[0]} = time steps")
    print(f"  {out.shape[1]} = batch size")
    print(f"  {out.shape[2]} = vocab size")
    print("\nModel ready.")
