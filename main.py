"""
Project 41: Text Classification with Recurrent Neural Networks (RNN)
=====================================================================
Description:
End-to-end binary sentiment classification on movie reviews (Stanford Sentiment Treebank /
Rotten Tomatoes) using an Elman Recurrent Neural Network (RNN) with learnable word embeddings.
Features custom vocabulary indexing, sequence padding and packing, bidirectional recurrence,
gradient clipping for BPTT stability, temporal hidden-state activation profiling, and
comprehensive ROC/F1-score evaluation.
"""

import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Configuration & Seed Initialization
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
EMBED_DIM = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
BATCH_SIZE = 64
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_SEQ_LEN = 60
MIN_WORD_FREQ = 2

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def set_seed(seed: int = 42) -> None:
    """Ensure strict deterministic reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Tokenizer & Vocabulary Construction
# -----------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Tokenizes text using alphanumeric word and punctuation regex."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    """Builds and manages word-to-index and index-to-word mappings."""
    def __init__(self, min_freq: int = MIN_WORD_FREQ):
        self.min_freq = min_freq
        self.w2i = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.i2w = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    def build_vocab(self, texts: List[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        idx = len(self.w2i)
        for word, count in counter.items():
            if count >= self.min_freq and word not in self.w2i:
                self.w2i[word] = idx
                self.i2w[idx] = word
                idx += 1

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        tokens = tokenize(text)[:max_len]
        return [self.w2i.get(token, UNK_IDX) for token in tokens]

    def __len__(self) -> int:
        return len(self.w2i)


# -----------------------------------------------------------------------------
# Dataset & Custom Collation
# -----------------------------------------------------------------------------
class MovieReviewDataset(Dataset):
    """Loads tab-separated movie reviews and binary sentiment labels."""
    def __init__(self, filepath: Path, vocab: Vocabulary, max_len: int = MAX_SEQ_LEN):
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text, label = parts[0], int(parts[1])
                    indices = vocab.encode(text, max_len=max_len)
                    if len(indices) > 0:
                        self.samples.append((torch.tensor(indices, dtype=torch.long), label, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        return self.samples[idx]


def collate_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Pads variable-length token sequences and sorts by length for packed RNN execution."""
    # Sort batch by sequence length in descending order
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels, texts = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_seqs, labels, lengths, list(texts)


# -----------------------------------------------------------------------------
# Recurrent Neural Network (RNN) Architecture
# -----------------------------------------------------------------------------
class SentimentRNN(nn.Module):
    """
    Bidirectional Elman Recurrent Neural Network with packed sequences.
    h_t = tanh(W_ih * e_t + b_ih + W_hh * h_{t-1} + b_{hh})
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Concatenate forward and backward final hidden states
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Embedding lookup: [B, T] -> [B, T, D]
        embedded = self.dropout(self.embedding(x))

        # 2. Packed Sequence for optimal recurrence (skips padding computations)
        packed_embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, hidden = self.rnn(packed_embedded)

        # 3. Hidden state concatenation: [num_layers * num_directions, B, H]
        # Top layer forward: hidden[-2], top layer backward: hidden[-1]
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        combined_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)  # [B, 2*H]

        logits = self.fc(combined_hidden).squeeze(1)

        if return_states:
            # Unpack full temporal sequence of hidden states for dynamic profiling
            unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
            return logits, unpacked_output
        return logits, None


# -----------------------------------------------------------------------------
# Training and Evaluation Engine
# -----------------------------------------------------------------------------
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Runs a single training epoch with gradient clipping to stabilize BPTT."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y, lengths, _ in dataloader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()
        logits, _ = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping to eliminate exploding gradients in recurrent backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += preds.eq(y).sum().item()
        total += x.size(0)

    return running_loss / total, (correct / total) * 100.0


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Computes comprehensive evaluation metrics on validation set."""
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for x, y, lengths, _ in dataloader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            logits, _ = model(x, lengths)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            running_loss += loss.item() * x.size(0)
            total += x.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / total
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    auc = roc_auc_score(y_true, y_prob)

    return val_loss, acc, prec, rec, f1, auc, y_true, y_pred


# -----------------------------------------------------------------------------
# Diagnostic Plotting Dashboard
# -----------------------------------------------------------------------------
def plot_rnn_dashboard(
    history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: SentimentRNN,
    sample_dataset: MovieReviewDataset,
    device: torch.device,
    save_path: Path,
) -> None:
    """Generates a 4-panel publication-grade evaluation dashboard."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Binary Cross-Entropy Loss Trajectory
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, history["train_loss"], "o-", color="#1f77b4", linewidth=2.2, label="Train Loss")
    ax_loss.plot(epochs, history["val_loss"], "s--", color="#d62728", linewidth=2.2, label="Validation Loss")
    ax_loss.set_title("(a) Binary Cross-Entropy Loss Progression", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.set_xticks(list(epochs))
    ax_loss.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_loss.grid(True, linestyle=":", alpha=0.6)

    # 2. Validation Accuracy & F1-Score Progression
    ax_acc = axes[0, 1]
    ax_acc.plot(epochs, history["val_acc"], "o-", color="#2ca02c", linewidth=2.2, label="Val Accuracy (%)")
    ax_acc.plot(epochs, history["val_f1"], "^--", color="#ff7f0e", linewidth=2.2, label="Val F1-Score (%)")
    ax_acc.set_title("(b) Validation Classification Metrics", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Score (%)", fontsize=11)
    ax_acc.set_xticks(list(epochs))
    ax_acc.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_acc.grid(True, linestyle=":", alpha=0.6)

    # 3. Recurrent Hidden State Dynamics (Temporal Norm Evolution)
    ax_dyn = axes[1, 0]
    model.eval()

    # Select representative positive and negative reviews
    pos_idx = next(i for i in range(len(sample_dataset)) if sample_dataset[i][1] == 1 and len(sample_dataset[i][0]) > 15)
    neg_idx = next(i for i in range(len(sample_dataset)) if sample_dataset[i][1] == 0 and len(sample_dataset[i][0]) > 15)

    with torch.no_grad():
        for sample_i, col, label_name in [(pos_idx, "#2ca02c", "Positive Review"), (neg_idx, "#d62728", "Negative Review")]:
            seq, lbl, raw_txt = sample_dataset[sample_i]
            x_in = seq.unsqueeze(0).to(device)
            lens = torch.tensor([len(seq)])
            _, unpacked_states = model(x_in, lens, return_states=True)
            # unpacked_states: [1, T, 2*H]
            states = unpacked_states.squeeze(0).cpu()
            # Calculate L2 norm of forward and backward hidden states across time steps
            norms = torch.norm(states, p=2, dim=1).numpy()
            time_steps = range(1, len(norms) + 1)
            ax_dyn.plot(time_steps, norms, "o-", color=col, linewidth=2.0, label=f"{label_name} ($L_2$ Norm)")

    ax_dyn.set_title("(c) Temporal Recurrent Hidden State Dynamics ($\|h_t\|_2$)", fontsize=13, fontweight="bold", pad=10)
    ax_dyn.set_xlabel("Sequence Time Step $t$ (Word Position)", fontsize=11)
    ax_dyn.set_ylabel("Hidden State Activation Magnitude $\|h_t\|_2$", fontsize=11)
    ax_dyn.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    ax_dyn.grid(True, linestyle=":", alpha=0.6)

    # 4. Confusion Matrix Heatmap
    ax_cm = axes[1, 1]
    cm = confusion_matrix(y_true, y_pred)
    im_cm = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    ax_cm.set_title("(d) Validation Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
    fig.colorbar(im_cm, ax=ax_cm, fraction=0.046, pad=0.04)

    classes = ["Negative", "Positive"]
    tick_marks = np.arange(len(classes))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_xticklabels(classes, fontsize=10)
    ax_cm.set_yticklabels(classes, fontsize=10)
    ax_cm.set_xlabel("Predicted Sentiment", fontsize=11)
    ax_cm.set_ylabel("Ground Truth Sentiment", fontsize=11)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / np.sum(cm[i, :]) * 100.0
            ax_cm.text(
                j, i, f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[Artifact] Saved RNN diagnostic dashboard to: {save_path}")


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main() -> None:
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("Project 41: Text Classification with Recurrent Neural Networks (RNN)")
    print("=" * 75)
    print(f"Compute Device        : {device}")
    print(f"Dataset Source        : Stanford Sentiment Treebank (SST-2 Movie Reviews)")
    print(f"Architecture          : 2-Layer Bidirectional Elman RNN with Tanh Activations")
    print(f"Embedding Dimension   : {EMBED_DIM} | Hidden Recurrent Dimension: {HIDDEN_DIM}")
    print(f"Max Sequence Length   : {MAX_SEQ_LEN} tokens | Batch Size: {BATCH_SIZE}")
    print(f"Epochs                : {NUM_EPOCHS} | Optimizer: AdamW (lr={LEARNING_RATE})")
    print("-" * 75)

    # 1. Build Vocabulary from Training Corpus
    train_path = DATA_DIR / "train.tsv"
    val_path = DATA_DIR / "dev.tsv"

    raw_texts = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                raw_texts.append(parts[0])

    vocab = Vocabulary(min_freq=MIN_WORD_FREQ)
    vocab.build_vocab(raw_texts)
    print(f"Constructed Vocabulary: {len(vocab):,} unique tokens (min_freq={MIN_WORD_FREQ})")

    # 2. Construct Datasets & Dataloaders
    train_dataset = MovieReviewDataset(train_path, vocab, max_len=MAX_SEQ_LEN)
    val_dataset = MovieReviewDataset(val_path, vocab, max_len=MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    print(f"Training Samples      : {len(train_dataset):,} reviews")
    print(f"Validation Samples    : {len(val_dataset):,} reviews")
    print("-" * 75)

    # 3. Model & Optimizer
    model = SentimentRNN(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters      : {total_params:,}")
    print("-" * 75)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    # 4. Training Loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    start_time = time.perf_counter()

    for epoch in range(1, NUM_EPOCHS + 1):
        ep_start = time.perf_counter()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc, y_true, y_pred = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)

        ep_duration = time.perf_counter() - ep_start
        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
            f"Train Loss: {t_loss:.4f} | Train Acc: {t_acc:6.2f}% | "
            f"Val Loss: {v_loss:.4f} | Val Acc: {v_acc:6.2f}% | "
            f"Val F1: {v_f1:6.2f}% | AUC: {v_auc:.4f} | Time: {ep_duration:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Training Completed in {total_time:.2f} seconds.")
    print(f"Final Validation Top-1 Accuracy: {history['val_acc'][-1]:.2f}% | Final F1: {history['val_f1'][-1]:.2f}%")

    # 5. Save Model Checkpoint
    checkpoint_path = CHECKPOINT_DIR / "rnn_sentiment_model.pt"
    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "val_acc": history["val_acc"][-1],
            "val_f1": history["val_f1"][-1],
        },
        checkpoint_path,
    )
    print(f"[Checkpoint] Saved trained RNN model to: {checkpoint_path}")

    # 6. Generate Publication-Quality Evaluation Figure
    figure_path = PROJECT_DIR / "rnn_text_classification_results.png"
    plot_rnn_dashboard(history, y_true, y_pred, model, val_dataset, device, figure_path)

    print("=" * 75)
    print("Execution pipeline finished successfully.")


if __name__ == "__main__":
    main()

