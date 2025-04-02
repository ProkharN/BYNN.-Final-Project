#!/usr/bin/env python
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# -----------------------------
# Global Settings
# -----------------------------
DATA_DIR = "./data"  # Adjust if needed
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# Set random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# Data Loading Utilities
# -----------------------------
def load_imdb_data(split="train"):
    """
    Loads IMDB data from the local folder structure:
      data/train/pos/*.txt
      data/train/neg/*.txt
      data/test/pos/*.txt
      data/test/neg/*.txt

    Returns a list of (text, label), where label is 1 for pos and 0 for neg.
    """
    data = []
    base_path = TRAIN_DIR if split == "train" else TEST_DIR
    
    for label_dir in ["pos", "neg"]:
        label = 1 if label_dir == "pos" else 0
        full_dir = os.path.join(base_path, label_dir)
        for fname in os.listdir(full_dir):
            if fname.endswith(".txt"):
                file_path = os.path.join(full_dir, fname)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                data.append((text, label))
    return data

def tokenize(text):
    """
    A very basic tokenizer splitting on whitespace and punctuation.
    You can customize or replace with something like spaCy, nltk, etc.
    """
    # Lowercase for consistency
    text = text.lower()
    # Split on whitespace
    tokens = text.split()
    return tokens

def build_vocab(dataset, min_freq=2):
    """
    Builds a vocabulary (word -> index) from the dataset.
    dataset: list of (text, label)
    min_freq: only include words that appear at least min_freq times
    """
    from collections import Counter
    word_counter = Counter()
    for text, _ in dataset:
        tokens = tokenize(text)
        word_counter.update(tokens)
    
    # Sort by frequency
    sorted_words = sorted([w for w, c in word_counter.items() if c >= min_freq],
                          key=lambda w: word_counter[w],
                          reverse=True)
    
    # Special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for w in sorted_words:
        vocab[w] = idx
        idx += 1
    return vocab

def text_to_tensor(text, vocab):
    """
    Converts text (string) to a list of vocab indices.
    """
    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    return torch.tensor(indices, dtype=torch.long)

# -----------------------------
# PyTorch Dataset and Collate
# -----------------------------
class IMDBDataset(Dataset):
    def __init__(self, data_list, vocab):
        """
        data_list: list of (text, label)
        vocab: dict mapping word -> index
        """
        self.data_list = data_list
        self.vocab = vocab

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        text, label = self.data_list[idx]
        indices = text_to_tensor(text, self.vocab)
        return indices, label

def collate_fn(batch):
    """
    Collate function to pad sequences and create a batch of data.
    batch: list of (indices_tensor, label)
    """
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    
    labels_tensor = torch.tensor(labels, dtype=torch.float)
    return texts_padded, labels_tensor, lengths

# -----------------------------
# Model Definitions
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, num_classes=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # Pack padded sequence for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), 
                                                   batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out.squeeze()

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, kernel_sizes=[3,4,5], num_filters=100, num_classes=1):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, lengths=None):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        conv_outs = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        out = self.dropout(cat)
        out = self.fc(out)
        return out.squeeze()

class FFClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=1):
        super(FFClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # Compute average embeddings, ignoring padding
        mask = (x != 0).unsqueeze(-1).float()
        summed = torch.sum(embedded * mask, dim=1)
        lengths = lengths.unsqueeze(1).float().clamp(min=1)
        averaged = summed / lengths
        out = torch.relu(self.fc1(averaged))
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze()

# -----------------------------
# Training and Evaluation
# -----------------------------
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    for texts, labels, lengths in dataloader:
        texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * texts.size(0)
        # Detach preds before converting to numpy
        preds = torch.round(torch.sigmoid(outputs))
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels, lengths in dataloader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * texts.size(0)
            preds = torch.round(torch.sigmoid(outputs))
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, precision, recall, f1


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data...")
    train_data = load_imdb_data(split="train")
    test_data = load_imdb_data(split="test")

    print("Building vocabulary...")
    vocab = build_vocab(train_data, min_freq=2)
    vocab_size = len(vocab)
    print("Vocabulary size:", vocab_size)

    # Create Datasets
    train_dataset = IMDBDataset(train_data, vocab)
    test_dataset = IMDBDataset(test_data, vocab)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_fn)

    # Model hyperparameters
    embed_dim = 128
    hidden_dim = 128
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 5

    # Define models
    models = {
        "LSTM": LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_layers).to(device),
        "CNN": CNNClassifier(vocab_size, embed_dim).to(device),
        "FeedForward": FFClassifier(vocab_size, embed_dim, hidden_dim).to(device)
    }

    criterion = nn.BCEWithLogitsLoss()
    optimizers = {name: optim.Adam(model.parameters(), lr=learning_rate) 
                  for name, model in models.items()}

    # Track performance
    performance = {
        name: {
            "Train Loss": [], "Train Acc": [], 
            "Test Loss": None, "Test Acc": None,
            "Precision": None, "Recall": None, "F1 Score": None
        } 
        for name in models.keys()
    }

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} model ---")
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, train_acc = train_model(model, train_loader, optimizers[name], criterion, device)
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Time: {elapsed:.2f}s")
            performance[name]["Train Loss"].append(train_loss)
            performance[name]["Train Acc"].append(train_acc)
        
        # Evaluate on test set
        test_loss, test_acc, precision, recall, f1 = evaluate_model(model, test_loader, criterion, device)
        performance[name]["Test Loss"] = test_loss
        performance[name]["Test Acc"] = test_acc
        performance[name]["Precision"] = precision
        performance[name]["Recall"] = recall
        performance[name]["F1 Score"] = f1

        print(f"{name} Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} "
              f"| Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Summarize results
    summary = []
    for name, metrics in performance.items():
        summary.append({
            "Model": name,
            "Test Loss": metrics["Test Loss"],
            "Test Acc": metrics["Test Acc"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"]
        })
    summary_df = pd.DataFrame(summary)
    print("\nSummary of Model Performance on Test Set:")
    print(summary_df)
    
    # Save results to CSV
    summary_df.to_csv("model_performance_summary.csv", index=False)
    print("Performance summary saved to model_performance_summary.csv")

if __name__ == "__main__":
    main()
