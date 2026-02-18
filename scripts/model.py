"""Phase 4: Model Architectures.

Three model types:
  1. MajorityClassBaseline - predicts training prevalence
  2. PerPathologyClassifier - TF-IDF + per-pathology LogisticRegression
  3. RadBERTClassifier - RadBERT encoder + linear classification head
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class MajorityClassBaseline:
    """Predicts training set prevalence as probability for every input.

    AUROC is 0.5 by definition (no discriminative power).
    """

    def __init__(self):
        self.prevalences = np.zeros(len(PATHOLOGIES))

    def fit(self, labels: np.ndarray, mask: np.ndarray | None = None):
        """Compute per-pathology prevalence from training labels."""
        for j in range(labels.shape[1]):
            if mask is not None:
                valid = mask[:, j] == 1.0
                col = labels[valid, j]
            else:
                col = labels[:, j]
            self.prevalences[j] = col.mean() if len(col) > 0 else 0.5

    def predict(self, n_samples: int) -> np.ndarray:
        """Return prevalence for all samples."""
        return np.tile(self.prevalences, (n_samples, 1))

    def save(self, path: str):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({"prevalences": self.prevalences}, f)

    @classmethod
    def load(cls, path: str) -> "MajorityClassBaseline":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls()
        model.prevalences = data["prevalences"]
        return model


class PerPathologyClassifier:
    """14 independent LogisticRegression classifiers on TF-IDF features.

    Handles per-pathology NaN masking for U-Ignore strategy.
    """

    def __init__(self, class_weight="balanced", max_iter=1000, solver="lbfgs"):
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver
        self.classifiers = {}

    def fit(
        self,
        X,
        labels: np.ndarray,
        mask: np.ndarray | None = None,
    ):
        """Train one LogisticRegression per pathology."""
        from tqdm import tqdm

        for j, pathology in enumerate(tqdm(PATHOLOGIES, desc="Training classifiers")):
            y = labels[:, j]

            if mask is not None:
                valid = mask[:, j] == 1.0
                X_train = X[valid]
                y_train = y[valid]
            else:
                X_train = X
                y_train = y

            # Skip if only one class present
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"  Warning: {pathology} has only class(es) {unique_classes}, skipping")
                self.classifiers[pathology] = None
                continue

            clf = LogisticRegression(
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                solver=self.solver,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            self.classifiers[pathology] = clf

    def predict_proba(self, X) -> np.ndarray:
        """Return probability of positive class for all pathologies."""
        n = X.shape[0]
        probs = np.zeros((n, len(PATHOLOGIES)))

        for j, pathology in enumerate(PATHOLOGIES):
            clf = self.classifiers.get(pathology)
            if clf is not None:
                probs[:, j] = clf.predict_proba(X)[:, 1]
            else:
                probs[:, j] = 0.5  # Default for missing classifiers

        return probs

    def save(self, path: str):
        """Save all classifiers to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "classifiers": self.classifiers,
                "class_weight": self.class_weight,
                "max_iter": self.max_iter,
                "solver": self.solver,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PerPathologyClassifier":
        """Load classifiers from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(
            class_weight=data["class_weight"],
            max_iter=data["max_iter"],
            solver=data["solver"],
        )
        model.classifiers = data["classifiers"]
        return model


class RadBERTClassifier(nn.Module):
    """RadBERT encoder with multi-label classification head.

    Architecture: RadBERT -> [CLS] pooling -> Dropout -> Linear(768, 14)
    Loss: BCEWithLogitsLoss with per-pathology pos_weight and label masking.
    """

    def __init__(
        self,
        model_name: str = "StanfordAIMI/RadBERT",
        num_labels: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for BERT-base
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked BCEWithLogitsLoss.

        label_mask: 1.0 for valid labels, 0.0 for masked (NaN) labels.
        """
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = loss_fn(logits, labels)
        # Apply mask — only compute loss for valid labels
        loss = loss * label_mask
        # Average over valid labels only
        n_valid = label_mask.sum()
        if n_valid > 0:
            loss = loss.sum() / n_valid
        else:
            loss = loss.sum() * 0.0  # No valid labels
        return loss
