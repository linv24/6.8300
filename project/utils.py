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
from scipy.interpolate import interp1d

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

def interpolate_to_length(values, target_length):
    """
    Interpolate values to target_length, for plotting loss curves that have
    different numbers of training steps
    """
    x_original = np.linspace(0, 1, len(values))
    x_target = np.linspace(0, 1, target_length)
    f = interp1d(x_original, values, kind="linear")
    return f(x_target).tolist()

def plot_metrics(predicates):
    max_length = 800
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].set_title("Averaged Training Loss")
    ax[0,1].set_title("Averaged Training Accuracy")
    ax[1,0].set_title("Averaged Validation Loss")
    ax[1,1].set_title("Averaged Validation Accuracy")

    cmap = plt.get_cmap("tab20", len(predicates))
    
    for ix, predicate in enumerate(predicates):
        all_metrics = torch.load(f"{OUTPUT_DIRECTORY}/trainval_metrics_{predicate}.pt", weights_only=False)

        all_train_losses = all_metrics["all_train_losses"]
        all_val_losses = all_metrics["all_val_losses"]
        all_train_accs = [[compute_cm_metrics(cm)["accuracy"] for cm in cm_list] for cm_list in all_metrics["all_train_confusion_matrices"]]
        all_val_accs = [[compute_cm_metrics(cm)["accuracy"] for cm in cm_list] for cm_list in all_metrics["all_val_confusion_matrices"]]

        interpolated_train_losses = interpolate_to_length(np.mean(all_train_losses, axis=0), max_length)
        interpolated_val_losses = interpolate_to_length(np.mean(all_val_losses, axis=0), max_length)
        interpolated_train_accs = interpolate_to_length(np.mean(all_train_accs, axis=0), max_length)
        interpolated_val_accs = interpolate_to_length(np.mean(all_val_accs, axis=0), max_length)
        x = np.linspace(0, 1, max_length)

        color = cmap(ix)
        ax[0,0].plot(x, interpolated_train_losses, color=color, label=predicate)
        ax[0,1].plot(x, interpolated_train_accs, color=color)
        ax[1,0].plot(x, interpolated_val_losses, color=color)
        ax[1,1].plot(x, interpolated_val_accs, color=color)

    fig.suptitle(f"Metrics for predicates")
    fig.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.show()

def evaluate(model, test_dataloader, embedding_type="image", 
             device="cuda", verbose=False):
    model.to(device)
    criterion = BCEWithLogitsLoss()
    
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            if embedding_type == "image":
                x = batch["image_emb"].to(device)
            elif embedding_type == "multimodal":
                # TODO
                raise NotImplementedError
            else:
                raise Exception
            y = batch["label"].float().to(device)
            logits = model(x)
            loss = criterion(logits, y)

            test_loss += loss.item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(y.cpu().long().tolist())

        test_loss /= len(test_labels)
        test_cm = confusion_matrix(test_preds, test_labels, labels=[0, 1])
        test_acc = sum(pred == label for pred, label in zip(test_preds, test_labels)) / len(test_preds)

    if verbose:
        print(f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
    return test_loss, test_cm

def get_val_metrics(predicates):
    """
    Returns validation clf metrics for the last training step.
    """
    d = defaultdict(dict)

    for ix, predicate in enumerate(predicates):
        # validation metrics
        all_metrics = torch.load(
            f"{OUTPUT_DIRECTORY}/trainval_metrics_{predicate}.pt", weights_only=False
        )
        all_val_cms = all_metrics["all_val_confusion_matrices"]
        best_val_cms = [cm_list[-1] for cm_list in all_val_cms] # only get last step's metrics
        best_val_metrics = [compute_cm_metrics(cm) for cm in best_val_cms]
        d[predicate] = {
            "accuracy": np.mean([metric["accuracy"] for metric in best_val_metrics]),
            "precision": np.mean([metric["precision"] for metric in best_val_metrics]),
            "recall": np.mean([metric["recall"] for metric in best_val_metrics]),
            "f1": np.mean([metric["f1"] for metric in best_val_metrics]),
        } 

    return d

def get_test_metrics(predicates):
    """
    Returns test clf metrics as a dict of the format:
        predicate: {
            accuracy: ...,
            precision: ...,
            recall: ...,
            f1: ...,
        }
    """
    d = defaultdict(dict)

    for predicate in predicates:
        data_file_path = f"{DATA_DIRECTORY}/predicate_probe_data/{predicate}"
        test_ds = PredicateProbeDataset(data_file_path + "_test.pt")
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False) 

        mlp_model = MLP()
        mlp_model.load_state_dict(torch.load(f"{MODEL_DIRECTORY}/mlp_{predicate}.pt"))
        test_loss, test_cm = evaluate(mlp_model, test_dl)
        d[predicate] = compute_cm_metrics(test_cm)

    return d

def plot_bar_clf_metrics(predicates, split):
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    num_predicates = len(predicates)
    num_metrics = len(metric_labels)

    if split == "val":
        metrics = get_val_metrics(predicates)
    elif split == "test":
        metrics = get_test_metrics(predicates)
    metrics_array = np.array(
        [[metric["accuracy"], metric["precision"], metric["recall"], metric["f1"]]
         for metric in metrics.values()]
    )

    # Bar settings
    x = np.arange(num_predicates)
    bar_width = 0.2
    offsets = np.arange(num_metrics) * bar_width

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(num_metrics):
        ax.bar(x + offsets[i], metrics_array[:, i], width=bar_width, label=metric_labels[i])

    # Ticks and labels
    ax.set_xticks(x + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(predicates, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Predicate-wise Evaluation Metrics for {split.capitalize()} Split")
    ax.legend()
    plt.tight_layout()
    plt.show()