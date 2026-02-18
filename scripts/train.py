"""Phase 4: Training Loop Orchestration.

Trains the 4×3 experiment matrix (4 uncertainty strategies × 3 model types).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.model import (
    PATHOLOGIES,
    MajorityClassBaseline,
    PerPathologyClassifier,
    RadBERTClassifier,
)


def load_class_weights(processed_dir: Path) -> dict:
    """Load precomputed class weights."""
    weights_path = processed_dir / "class_weights.json"
    with open(weights_path) as f:
        return json.load(f)


def train_naive(
    strategy: str,
    processed_dir: Path,
    models_dir: Path,
):
    """Train MajorityClassBaseline."""
    print(f"\n--- Naive Baseline ({strategy}) ---")
    labels = np.load(processed_dir / f"labels_{strategy}_train.npy")
    mask = np.load(processed_dir / f"mask_{strategy}_train.npy")

    model = MajorityClassBaseline()
    model.fit(labels, mask)

    model_path = models_dir / f"naive_{strategy}.pkl"
    model.save(str(model_path))
    print(f"  Prevalences: {dict(zip(PATHOLOGIES, model.prevalences.round(4)))}")
    print(f"  Saved to {model_path}")


def train_classical(
    strategy: str,
    processed_dir: Path,
    models_dir: Path,
    config: dict,
):
    """Train PerPathologyClassifier with TF-IDF features."""
    print(f"\n--- Classical ML ({strategy}) ---")

    X_train = load_npz(processed_dir / "tfidf_train.npz")
    labels = np.load(processed_dir / f"labels_{strategy}_train.npy")
    mask = np.load(processed_dir / f"mask_{strategy}_train.npy")

    classical_config = config["classical"]
    model = PerPathologyClassifier(
        class_weight=classical_config["class_weight"],
        max_iter=classical_config["max_iter"],
        solver=classical_config["solver"],
    )
    model.fit(X_train, labels, mask)

    model_path = models_dir / f"classical_{strategy}.pkl"
    model.save(str(model_path))
    print(f"  Saved to {model_path}")


def train_radbert(
    strategy: str,
    processed_dir: Path,
    models_dir: Path,
    config: dict,
):
    """Train RadBERTClassifier with fine-tuning."""
    import pandas as pd
    from scripts.build_features import get_radbert_dataset_class

    print(f"\n--- RadBERT ({strategy}) ---")
    radbert_config = config["radbert"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load data
    df = pd.read_parquet(processed_dir / f"labels_{strategy}.parquet")
    train_labels = np.load(processed_dir / f"labels_{strategy}_train.npy")
    train_mask = np.load(processed_dir / f"mask_{strategy}_train.npy")
    val_labels = np.load(processed_dir / f"labels_{strategy}_val.npy")
    val_mask = np.load(processed_dir / f"mask_{strategy}_val.npy")

    train_texts = df[df["split"] == "train"]["report_text"].tolist()
    val_texts = df[df["split"] == "val"]["report_text"].tolist()

    # Create datasets
    RadBERTDataset = get_radbert_dataset_class()
    train_dataset = RadBERTDataset(
        texts=train_texts,
        labels=train_labels,
        label_mask=train_mask,
        tokenizer_name=radbert_config["model_name"],
        max_length=radbert_config["max_length"],
    )
    val_dataset = RadBERTDataset(
        texts=val_texts,
        labels=val_labels,
        label_mask=val_mask,
        tokenizer_name=radbert_config["model_name"],
        max_length=radbert_config["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=radbert_config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=radbert_config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Initialize model
    model = RadBERTClassifier(
        model_name=radbert_config["model_name"],
        num_labels=len(PATHOLOGIES),
        dropout=radbert_config["dropout"],
    ).to(device)

    # Class weights for loss
    class_weights = load_class_weights(processed_dir)
    pos_weight = torch.tensor(
        [class_weights.get(p, 1.0) for p in PATHOLOGIES],
        dtype=torch.float32,
    ).to(device)
    # Clip extreme weights
    pos_weight = pos_weight.clamp(max=100.0)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=radbert_config["learning_rate"],
        weight_decay=radbert_config["weight_decay"],
    )

    num_training_steps = len(train_loader) * radbert_config["num_epochs"]
    num_warmup_steps = int(num_training_steps * radbert_config["warmup_ratio"])

    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )
    decay_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[num_warmup_steps],
    )

    # Mixed precision
    use_fp16 = radbert_config["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    # Training loop
    best_val_auroc = 0.0
    patience_counter = 0
    patience = radbert_config["early_stopping_patience"]

    for epoch in range(radbert_config["num_epochs"]):
        # Train
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{radbert_config['num_epochs']} [Train]")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            label_mask = batch["label_mask"].to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = model.compute_loss(logits, labels, label_mask, pos_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask)
                loss = model.compute_loss(logits, labels, label_mask, pos_weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        all_masks = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                label_mask = batch["label_mask"].to(device)

                if use_fp16:
                    with torch.amp.autocast("cuda"):
                        logits = model(input_ids, attention_mask)
                        loss = model.compute_loss(logits, labels, label_mask, pos_weight)
                else:
                    logits = model(input_ids, attention_mask)
                    loss = model.compute_loss(logits, labels, label_mask, pos_weight)

                val_losses.append(loss.item())
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_masks.append(label_mask.cpu())

        avg_val_loss = np.mean(val_losses)

        # Compute macro-AUROC on validation
        from sklearn.metrics import roc_auc_score

        all_logits = torch.cat(all_logits).sigmoid().numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_masks = torch.cat(all_masks).numpy()

        aurocs = []
        for j in range(len(PATHOLOGIES)):
            valid = all_masks[:, j] == 1.0
            y_true = all_labels[valid, j]
            y_pred = all_logits[valid, j]
            if len(np.unique(y_true)) >= 2:
                aurocs.append(roc_auc_score(y_true, y_pred))
        macro_auroc = np.mean(aurocs) if aurocs else 0.0

        print(f"  Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, macro_auroc={macro_auroc:.4f}")

        # Early stopping
        if macro_auroc > best_val_auroc:
            best_val_auroc = macro_auroc
            patience_counter = 0
            # Save best model
            model_path = models_dir / f"radbert_{strategy}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved (AUROC={macro_auroc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    print(f"  Best validation macro-AUROC: {best_val_auroc:.4f}")


def run_training(
    config: dict,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
):
    """Run the full training pipeline."""
    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    if strategies is None:
        strategies = config["strategies"]
    if models is None:
        models = config["models"]

    print(f"Training {len(models)} model types × {len(strategies)} strategies")
    print(f"Models: {models}")
    print(f"Strategies: {strategies}")

    total_start = time.time()

    for strategy in strategies:
        # Check if label files exist
        labels_path = processed_dir / f"labels_{strategy}_train.npy"
        if not labels_path.exists():
            print(f"\nSkipping {strategy}: label files not found")
            continue

        for model_type in models:
            start = time.time()

            if model_type == "naive":
                train_naive(strategy, processed_dir, models_dir)
            elif model_type == "classical":
                train_classical(strategy, processed_dir, models_dir, config)
            elif model_type == "radbert":
                train_radbert(strategy, processed_dir, models_dir, config)

            elapsed = time.time() - start
            print(f"  Time: {elapsed / 60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\nTotal training time: {total_elapsed / 60:.1f} min")
    print("Phase 4 complete!")


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    run_training(config)
