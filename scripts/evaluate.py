"""Phase 5: Evaluation & Experiment Comparison.

Computes metrics (AUROC, AUPRC, F1, etc.) across the 4×3 experiment matrix.
Generates tables, plots, and error analysis.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.sparse import load_npz
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from scripts.model import (
    PATHOLOGIES,
    MajorityClassBaseline,
    PerPathologyClassifier,
    RadBERTClassifier,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute per-pathology and aggregate metrics."""
    results = {"per_pathology": {}, "aggregate": {}}

    aurocs = []
    auprcs = []
    f1s = []

    for j, pathology in enumerate(PATHOLOGIES):
        valid = mask[:, j] == 1.0
        y_t = y_true[valid, j]
        y_p = y_pred[valid, j]

        metrics = {}

        # Skip if only one class
        if len(np.unique(y_t)) < 2:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
            metrics["f1"] = float("nan")
            metrics["precision"] = float("nan")
            metrics["recall"] = float("nan")
            metrics["specificity"] = float("nan")
            metrics["n_valid"] = int(valid.sum())
            metrics["n_positive"] = int(y_t.sum())
            results["per_pathology"][pathology] = metrics
            continue

        # AUROC
        try:
            metrics["auroc"] = float(roc_auc_score(y_t, y_p))
            aurocs.append(metrics["auroc"])
        except ValueError:
            metrics["auroc"] = float("nan")

        # AUPRC
        try:
            metrics["auprc"] = float(average_precision_score(y_t, y_p))
            auprcs.append(metrics["auprc"])
        except ValueError:
            metrics["auprc"] = float("nan")

        # Binary predictions at threshold
        y_binary = (y_p >= threshold).astype(int)

        metrics["f1"] = float(f1_score(y_t, y_binary, zero_division=0))
        f1s.append(metrics["f1"])
        metrics["precision"] = float(precision_score(y_t, y_binary, zero_division=0))
        metrics["recall"] = float(recall_score(y_t, y_binary, zero_division=0))

        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_t, y_binary, labels=[0, 1]).ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics["n_valid"] = int(valid.sum())
        metrics["n_positive"] = int(y_t.sum())

        results["per_pathology"][pathology] = metrics

    # Aggregate
    results["aggregate"]["macro_auroc"] = float(np.nanmean(aurocs)) if aurocs else float("nan")
    results["aggregate"]["weighted_auroc"] = _weighted_auroc(y_true, y_pred, mask)
    results["aggregate"]["macro_auprc"] = float(np.nanmean(auprcs)) if auprcs else float("nan")
    results["aggregate"]["macro_f1"] = float(np.nanmean(f1s)) if f1s else float("nan")

    return results


def _weighted_auroc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """Compute weighted AUROC (weighted by number of valid samples per pathology)."""
    total_weight = 0
    weighted_sum = 0
    for j in range(len(PATHOLOGIES)):
        valid = mask[:, j] == 1.0
        y_t = y_true[valid, j]
        y_p = y_pred[valid, j]
        if len(np.unique(y_t)) >= 2:
            w = valid.sum()
            weighted_sum += w * roc_auc_score(y_t, y_p)
            total_weight += w
    return float(weighted_sum / total_weight) if total_weight > 0 else float("nan")


def get_predictions(
    model_type: str,
    strategy: str,
    processed_dir: Path,
    models_dir: Path,
    config: dict,
) -> np.ndarray:
    """Get test set predictions for a given model/strategy combination."""
    if model_type == "naive":
        model = MajorityClassBaseline.load(str(models_dir / f"naive_{strategy}.pkl"))
        test_labels = np.load(processed_dir / f"labels_{strategy}_test.npy")
        return model.predict(test_labels.shape[0])

    elif model_type == "classical":
        model = PerPathologyClassifier.load(str(models_dir / f"classical_{strategy}.pkl"))
        X_test = load_npz(processed_dir / "tfidf_test.npz")
        return model.predict_proba(X_test)

    elif model_type == "radbert":
        model_path = models_dir / f"radbert_{strategy}.pt"
        if not model_path.exists():
            print(f"  Warning: {model_path} not found")
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RadBERTClassifier(
            model_name=config["radbert"]["model_name"],
            num_labels=len(PATHOLOGIES),
            dropout=config["radbert"]["dropout"],
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()

        # Load test data
        df = pd.read_parquet(processed_dir / f"labels_{strategy}.parquet")
        test_texts = df[df["split"] == "test"]["report_text"].tolist()

        from scripts.build_features import get_radbert_dataset_class

        test_labels = np.load(processed_dir / f"labels_{strategy}_test.npy")
        test_mask = np.load(processed_dir / f"mask_{strategy}_test.npy")

        RadBERTDataset = get_radbert_dataset_class()
        test_dataset = RadBERTDataset(
            texts=test_texts,
            labels=test_labels,
            label_mask=test_mask,
            tokenizer_name=config["radbert"]["model_name"],
            max_length=config["radbert"]["max_length"],
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["radbert"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )

        all_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="RadBERT inference"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    return None


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    strategy: str,
    output_dir: Path,
):
    """Plot ROC curves for all 14 pathologies."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for j, pathology in enumerate(PATHOLOGIES):
        ax = axes[j]
        valid = mask[:, j] == 1.0
        y_t = y_true[valid, j]
        y_p = y_pred[valid, j]

        if len(np.unique(y_t)) >= 2:
            fpr, tpr, _ = roc_curve(y_t, y_p)
            auroc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"AUROC={auroc:.3f}")
        else:
            ax.text(0.5, 0.5, "Single class", ha="center", va="center")

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(pathology, fontsize=10)
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for j in range(len(PATHOLOGIES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"ROC Curves: {model_name} / {strategy}", fontsize=14)
    plt.tight_layout()
    output_path = output_dir / f"roc_{model_name}_{strategy}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC curves to {output_path}")


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    model_name: str,
    strategy: str,
    output_dir: Path,
    threshold: float = 0.5,
):
    """Plot confusion matrices for all 14 pathologies."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    for j, pathology in enumerate(PATHOLOGIES):
        ax = axes[j]
        valid = mask[:, j] == 1.0
        y_t = y_true[valid, j]
        y_p = (y_pred[valid, j] >= threshold).astype(int)

        if len(np.unique(y_t)) >= 2:
            cm = confusion_matrix(y_t, y_p, labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        else:
            ax.text(0.5, 0.5, "Single class", ha="center", va="center")

        ax.set_title(pathology, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(len(PATHOLOGIES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Confusion Matrices: {model_name} / {strategy}", fontsize=14)
    plt.tight_layout()
    output_path = output_dir / f"cm_{model_name}_{strategy}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices to {output_path}")


def bootstrap_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for macro-AUROC."""
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    aurocs = []

    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        m = mask[indices]

        per_path = []
        for j in range(len(PATHOLOGIES)):
            valid = m[:, j] == 1.0
            yt = y_t[valid, j]
            yp = y_p[valid, j]
            if len(np.unique(yt)) >= 2:
                per_path.append(roc_auc_score(yt, yp))
        if per_path:
            aurocs.append(np.mean(per_path))

    return {
        "mean": float(np.mean(aurocs)),
        "std": float(np.std(aurocs)),
        "ci_lower": float(np.percentile(aurocs, 2.5)),
        "ci_upper": float(np.percentile(aurocs, 97.5)),
    }


def error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    n_examples: int = 5,
    threshold: float = 0.5,
) -> list[dict]:
    """Find the most confident mispredictions for error analysis."""
    errors = []

    for i in range(y_true.shape[0]):
        for j in range(len(PATHOLOGIES)):
            if mask[i, j] == 0:
                continue
            pred_binary = 1 if y_pred[i, j] >= threshold else 0
            if pred_binary != y_true[i, j]:
                confidence = abs(y_pred[i, j] - threshold)
                errors.append({
                    "index": i,
                    "pathology": PATHOLOGIES[j],
                    "true_label": int(y_true[i, j]),
                    "predicted_prob": float(y_pred[i, j]),
                    "confidence": confidence,
                    "report_text": df.iloc[i]["report_text"][:500],
                    "study_key": df.iloc[i]["study_key"],
                    "kaggle_image_dir": df.iloc[i].get("kaggle_image_dir", ""),
                })

    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors[:n_examples]


def run_evaluation(config: dict):
    """Run the full evaluation pipeline."""
    processed_dir = Path(config["data"]["processed_dir"])
    models_dir = Path("models")
    output_dir = Path(config["data"]["outputs_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = config["strategies"]
    model_types = config["models"]
    threshold = config["evaluation"]["threshold"]

    # Results table: model_type x strategy -> macro_auroc
    results_table = {}
    all_results = {}

    for strategy in strategies:
        test_labels_path = processed_dir / f"labels_{strategy}_test.npy"
        if not test_labels_path.exists():
            print(f"Skipping {strategy}: test labels not found")
            continue

        test_labels = np.load(test_labels_path)
        test_mask = np.load(processed_dir / f"mask_{strategy}_test.npy")

        for model_type in model_types:
            key = f"{model_type}_{strategy}"
            model_path = models_dir / f"{key}.pkl"
            if model_type == "radbert":
                model_path = models_dir / f"{key}.pt"
            if not model_path.exists():
                print(f"Skipping {key}: model not found")
                continue

            print(f"\n=== Evaluating {model_type} / {strategy} ===")

            # Get predictions
            y_pred = get_predictions(model_type, strategy, processed_dir, models_dir, config)
            if y_pred is None:
                continue

            # Compute metrics
            metrics = compute_metrics(test_labels, y_pred, test_mask, threshold)
            all_results[key] = metrics

            macro_auroc = metrics["aggregate"]["macro_auroc"]
            if strategy not in results_table:
                results_table[strategy] = {}
            results_table[strategy][model_type] = macro_auroc

            print(f"  Macro-AUROC: {macro_auroc:.4f}")

            # Plot ROC curves and confusion matrices for non-naive models
            if model_type != "naive":
                plot_roc_curves(test_labels, y_pred, test_mask,
                                model_type, strategy, output_dir)
                plot_confusion_matrices(test_labels, y_pred, test_mask,
                                        model_type, strategy, output_dir, threshold)

    # Print results table
    print("\n" + "=" * 60)
    print("4×3 Experiment Results (Macro-AUROC)")
    print("=" * 60)
    header = f"{'Strategy':<15}"
    for mt in model_types:
        header += f"{mt:>12}"
    print(header)
    print("-" * 60)
    for strategy in strategies:
        if strategy not in results_table:
            continue
        row = f"{strategy:<15}"
        for mt in model_types:
            val = results_table[strategy].get(mt, float("nan"))
            row += f"{val:>12.4f}"
        print(row)
    print("=" * 60)

    # Save results
    results_path = output_dir / "experiment_results.json"
    # Convert numpy types for JSON serialization
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved detailed results to {results_path}")

    # Save results table as CSV
    table_df = pd.DataFrame(results_table).T
    table_df.index.name = "strategy"
    table_path = output_dir / "results_table.csv"
    table_df.to_csv(table_path)
    print(f"Saved results table to {table_path}")

    # Bootstrap significance for best model
    print("\n=== Bootstrap Confidence Intervals ===")
    best_key = max(
        (k for k in all_results if "aggregate" in all_results[k]),
        key=lambda k: all_results[k]["aggregate"]["macro_auroc"],
        default=None,
    )
    if best_key:
        parts = best_key.split("_", 1)
        model_type, strategy = parts[0], parts[1]
        print(f"Best model: {best_key} (AUROC={all_results[best_key]['aggregate']['macro_auroc']:.4f})")

        test_labels = np.load(processed_dir / f"labels_{strategy}_test.npy")
        test_mask = np.load(processed_dir / f"mask_{strategy}_test.npy")
        y_pred = get_predictions(model_type, strategy, processed_dir, models_dir, config)

        if y_pred is not None:
            bs = bootstrap_auroc(
                test_labels, y_pred, test_mask,
                n_iterations=config["evaluation"]["bootstrap_n_iterations"],
                seed=config["evaluation"]["bootstrap_seed"],
            )
            print(f"  Bootstrap macro-AUROC: {bs['mean']:.4f} "
                  f"[{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")

    # Error analysis for best non-naive model
    print("\n=== Error Analysis ===")
    best_nontrivial = max(
        (k for k in all_results if not k.startswith("naive") and "aggregate" in all_results[k]),
        key=lambda k: all_results[k]["aggregate"]["macro_auroc"],
        default=None,
    )
    if best_nontrivial:
        parts = best_nontrivial.split("_", 1)
        model_type, strategy = parts[0], parts[1]
        print(f"Analyzing errors for: {best_nontrivial}")

        df = pd.read_parquet(processed_dir / f"labels_{strategy}.parquet")
        df_test = df[df["split"] == "test"].reset_index(drop=True)
        test_labels = np.load(processed_dir / f"labels_{strategy}_test.npy")
        test_mask = np.load(processed_dir / f"mask_{strategy}_test.npy")
        y_pred = get_predictions(model_type, strategy, processed_dir, models_dir, config)

        if y_pred is not None:
            errors = error_analysis(
                test_labels, y_pred, test_mask, df_test,
                n_examples=config["evaluation"]["n_error_examples"],
                threshold=threshold,
            )

            for i, err in enumerate(errors):
                print(f"\n  Error {i + 1}:")
                print(f"    Pathology: {err['pathology']}")
                print(f"    True: {err['true_label']}, Predicted: {err['predicted_prob']:.3f}")
                print(f"    Study: {err['study_key']}")
                print(f"    Kaggle image: {err['kaggle_image_dir']}")
                print(f"    Report: {err['report_text'][:200]}...")

            # Save error analysis
            errors_path = output_dir / "error_analysis.json"
            with open(errors_path, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"\n  Saved error analysis to {errors_path}")

    print("\nPhase 5 complete!")


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    run_evaluation(config)
