import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
import numpy as np

from config import *


class PredicateProbeDataset(Dataset):
    def __init__(self, pt_file_path):
        """
        pt_file_path should have the following format:
            <predicate>_<train/val/test>.pt

        Each sample should have the following keys:
            image_emb: rgb image embedding
            subject_emb: text embedding of subject
            object_emb: text embedding of object
            predicate_emb: text embedding of predicate
            depth_emb: depth image embedding
            subject_name: subject text
            object_name: object text
            predicate_name: predicate text
            weight: Rel3d sample weight
            label: True or False, for whether predicate holds
        """
        self.samples = torch.load(pt_file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, ix):
        return self.samples[ix]

class MLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, dropout=0.1):
        """
        CLIP uses 512-dimensional embeddings.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # outputs logits 
    
def train(model, train_dataloader, val_dataloader, embedding_type="image", 
          num_epochs=10, lr=1e-4, weight_decay=1e-5, device="cuda", verbose=False):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    criterion = BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    train_confusion_matrices = []
    val_confusion_matrices = []

    for epoch in tqdm(range(num_epochs), disable=not verbose):
        # training 
        model.train()
        epoch_train_loss = 0.0
        train_acc = 0.0
        train_preds = []
        train_labels = []

        for batch in train_dataloader:
            if embedding_type == "image":
                x = batch["image_emb"].to(device)
            elif embedding_type == "multimodal":
                # TODO
                pass
            else:
                raise Exception
            y = batch["label"].float().to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(y.cpu().long().tolist())

        train_cm = confusion_matrix(train_preds, train_labels, labels=[0, 1])
        train_acc = sum(pred == label for pred, label in zip(train_preds, train_labels)) / len(train_preds)

        train_losses.append(epoch_train_loss / len(train_labels))
        train_confusion_matrices.append(train_cm)

        # validation 
        model.eval()
        epoch_val_loss = 0.0
        val_acc = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                if embedding_type == "image":
                    x = batch["image_emb"].to(device)
                elif embedding_type == "multimodal":
                    # TODO
                    pass
                else:
                    raise Exception
                y = batch["label"].float().to(device)
                logits = model(x)
                loss = criterion(logits, y)

                epoch_val_loss += loss.item() * x.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(y.cpu().long().tolist())

            val_cm = confusion_matrix(val_preds, val_labels, labels=[0, 1])
            val_acc = sum(pred == label for pred, label in zip(val_preds, val_labels)) / len(val_preds)

            val_losses.append(epoch_val_loss / len(val_labels))
            val_confusion_matrices.append(val_cm)
        if verbose:
            print(f"  [epoch {epoch+1}] train/val loss: {epoch_train_loss/len(train_labels):.4f}/{epoch_val_loss/len(val_labels):.4f}, "
                  f"train/val acc: {train_acc:.4f}/{val_acc:.4f}")

    if verbose:
        print(f"Final train/val loss: {train_losses[-1]:.4f}/{val_losses[-1]:.4f}, "
            f"final train/val acc: {train_acc:.4f}/{val_acc:.4f}")
    return train_losses, val_losses, train_confusion_matrices, val_confusion_matrices
    
def compute_cm_metrics(confusion_matrix: np.ndarray, eps=1e-8):
    """
    Compute accuracy, precision, recall, and f1 from a confusion matrix.
    """
    assert confusion_matrix.shape == (2,2)

    tn, fp, fn, tp = confusion_matrix.flatten()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }