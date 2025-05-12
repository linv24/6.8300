from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from dataloader import create_dataloader, SpatialDataset
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
import numpy as np
import json
import time

from utils import *
from config import * 


device = "cuda" if torch.cuda.is_available() else "cpu"



def main():
    # training all predicates
    batch_size = 16
    total_steps = 8_000
    hidden_dim = 128
    num_trials = 5

    for predicate in allowed_predicates:
        print(f"starting mlp_{predicate}...")
        start_time = time.perf_counter()

        data_file_path = f"{DATA_DIRECTORY}/predicate_probe_data/{predicate}"
        train_ds = PredicateProbeDataset(data_file_path + "_train.pt")
        val_ds = PredicateProbeDataset(data_file_path + "_valid.pt")

        num_epochs = total_steps // (len(train_ds) // batch_size)

        best_val_acc = 0.0
        all_metrics = defaultdict(list)
        """
        all_metrics has the following keys:
            all_train_losses: num_trials lists of per-epoch training losses
            all_val_losses: num_trials lists of per-epoch val losses
            all_train_confusion_matrices: num_trials lists of per-epoch train cms
            all_val_confusion_matrices: num_trials lists of per-epoch val cms
        """

        for _ in range(num_trials):
            mlp_model = MLP(hidden_dim=hidden_dim)
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True) 
            val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True) 

            train_outputs = train(mlp_model, train_dl, val_dl, 
                                num_epochs=num_epochs, lr=1e-5, verbose=False)
            train_losses, val_losses, train_confusion_matrices, val_confusion_matrices = train_outputs
            all_metrics["all_train_losses"].append(train_losses)
            all_metrics["all_val_losses"].append(val_losses)
            all_metrics["all_train_confusion_matrices"].append(train_confusion_matrices)
            all_metrics["all_val_confusion_matrices"].append(val_confusion_matrices)

            last_val_acc = compute_cm_metrics(val_confusion_matrices[-1])["accuracy"]
            # save best model, based on validation accuracy
            if last_val_acc > best_val_acc:
                best_val_acc = last_val_acc
                torch.save(mlp_model.state_dict(), f"{MODEL_DIRECTORY}/mlp_{predicate}.pt")

        # save training metrics
        torch.save(all_metrics, f"{OUTPUT_DIRECTORY}/trainval_metrics_{predicate}.pt")

        print(f"finished mlp_{predicate} ({time.perf_counter() - start_time:.2f}s)")


if __name__ == "__main__":
    main()