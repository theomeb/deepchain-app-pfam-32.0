"""
Module that provide a classifier template to train a model on embeddings in
order to predict the family of a given protein.
The model is built with pytorch_ligthning, a wrapper on top of
pytorch (similar to keras with tensorflow)
"""

from biodatasets import load_dataset
from deepchain.models.utils import (
    dataloader_from_numpy,
)
from deepchain.models.torch_model import TorchModel

import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning.metrics.functional import accuracy

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from typing import Tuple


# Our custom protein family prediction MLP class
class FamilyMLP(TorchModel):
    """Multi-layer perceptron model."""

    def __init__(self, input_shape: int = 768, output_shape: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.output = nn.Softmax if output_shape > 1 else nn.Sigmoid
        self.loss = F.cross_entropy if output_shape > 1 else F.binary_cross_entropy
        self._model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_shape)
        )

    def forward(self, x):
        """Defines forward pass"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x).float()
        return self._model(x)

    def training_step(self, batch, batch_idx):
        """training_step defined the train loop. It is independent of forward"""
        x, y = batch
        y_hat = self._model(x)
        y = y.long()
        # y = torch.unsqueeze(y, 1)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self._model(x)
        y = y.long()

        loss = self.loss(y_hat, y)

        preds = torch.max(y_hat, dim=1)[1]
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def save_model(self, path: str):
        """Save entire model with torch"""
        torch.save(self._model, path)


# Load pfam dataset
pfam_dataset = load_dataset("pfam-32.0", force=True)
_, y = pfam_dataset.to_npy_arrays(input_names=["sequence"], target_names=["family_id"])

# Get embeddings and filter on available embeddings
embeddings = pfam_dataset.get_embeddings("sequence", "protbert", "mean")
available_embeddings_len = len(embeddings)
print(f"We take only the first {available_embeddings_len} sequences as we have only their embeddings available.")
y = y[0][:available_embeddings_len]

# Process targets
unique_classes = np.unique(y)
num_classes = len(unique_classes)
print(f"There are {num_classes} unique classes for family_id.")

# Encode target classes (families)
le = preprocessing.LabelEncoder()
labels = le.fit(unique_classes)
targets = le.transform(y)
print(f"Targets: {targets.shape}, {targets}, {len(labels.classes_)} classes")

# Load dataloaders
X_train, X_val, y_train, y_val = train_test_split(embeddings, targets, test_size=0.3)
train_dataloader = dataloader_from_numpy(X_train, y_train, batch_size=256)
val_dataloader = dataloader_from_numpy(X_val, y_val, batch_size=256)

# Fit the model and save it
mlp = FamilyMLP(input_shape=X_train.shape[1], output_shape=num_classes)
mlp.fit(train_dataloader, val_dataloader, epochs=100, auto_lr_find=True, auto_scale_batch_size=True)
mlp.save_model("family_model.pt")


# Model evaluation
def model_evaluation_accuracy(dataloader: DataLoader, model) -> Tuple[np.array, np.array]:
    """
    Make prediction for test data.
    Args:
        dataloader: a torch dataloader containing dataset to be evaluated
        model : a callable trained model with a predict method
    """
    y_pred, y_truth = [], []
    for X, y in dataloader:
        y_hat = torch.max(model.predict(X), 1)[1]
        y_pred += y_hat
        y_truth += y.detach().numpy().flatten().tolist()

    y_pred, y_truth = np.array(y_pred), np.array(y_truth)

    acc_score = accuracy_score(y_truth, y_pred)
    print(f" Test :  accuracy score : {acc_score:0.2f}")

    return y_pred, y_truth


prediction, truth = model_evaluation_accuracy(train_dataloader, mlp)
