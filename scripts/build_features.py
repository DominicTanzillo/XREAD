"""Phase 3: Feature Engineering.

TF-IDF vectorization for classical ML and RadBERT tokenization for deep learning.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


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


def build_tfidf(df: pd.DataFrame, config: dict, output_dir: Path) -> dict:
    """Fit TF-IDF vectorizer on training data and transform all splits.

    Returns dict with sparse matrices per split.
    """
    tfidf_config = config["tfidf"]

    # Fit on training data only
    train_mask = df["split"] == "train"
    train_texts = df.loc[train_mask, "report_text"].tolist()

    print(f"Fitting TF-IDF on {len(train_texts):,} training documents...")
    vectorizer = TfidfVectorizer(
        max_features=tfidf_config["max_features"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        sublinear_tf=tfidf_config["sublinear_tf"],
        strip_accents="unicode",
        stop_words="english",
        dtype=np.float32,
    )
    vectorizer.fit(train_texts)

    # Save vectorizer
    vec_path = Path(tfidf_config["vectorizer_path"])
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved vectorizer to {vec_path}")
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # Transform all splits
    results = {}
    for split in ["train", "val", "test"]:
        mask = df["split"] == split
        texts = df.loc[mask, "report_text"].tolist()
        X = vectorizer.transform(texts)
        results[split] = X

        # Save sparse matrix
        sparse_path = output_dir / f"tfidf_{split}.npz"
        save_npz(sparse_path, X)
        print(f"  {split}: {X.shape[0]:,} docs × {X.shape[1]:,} features -> {sparse_path}")

    return results


def get_radbert_dataset_class():
    """Return a PyTorch Dataset class for RadBERT tokenization.

    Tokenizes on-the-fly in __getitem__ to avoid massive disk usage.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    class RadBERTDataset(Dataset):
        """PyTorch Dataset that tokenizes reports on-the-fly with RadBERT tokenizer."""

        def __init__(
            self,
            texts: list[str],
            labels: np.ndarray,
            label_mask: np.ndarray | None = None,
            tokenizer_name: str = "StanfordAIMI/RadBERT",
            max_length: int = 512,
        ):
            self.texts = texts
            self.labels = torch.tensor(labels, dtype=torch.float32)
            if label_mask is not None:
                self.label_mask = torch.tensor(label_mask, dtype=torch.float32)
            else:
                self.label_mask = torch.ones_like(self.labels)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": self.labels[idx],
                "label_mask": self.label_mask[idx],
            }

    return RadBERTDataset


def prepare_labels_and_mask(
    df: pd.DataFrame,
    split: str,
    pathologies: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract label array and mask array for a given split.

    NaN labels get mask=0 (excluded from loss).
    """
    mask = df["split"] == split
    df_split = df[mask]

    labels = np.zeros((len(df_split), len(pathologies)), dtype=np.float32)
    label_mask = np.ones((len(df_split), len(pathologies)), dtype=np.float32)

    for j, p in enumerate(pathologies):
        if p in df_split.columns:
            vals = df_split[p].values
            for i in range(len(vals)):
                if pd.isna(vals[i]):
                    labels[i, j] = 0.0  # Placeholder
                    label_mask[i, j] = 0.0  # Mask out
                else:
                    labels[i, j] = float(vals[i])
                    label_mask[i, j] = 1.0

    return labels, label_mask


def run_feature_engineering(config: dict):
    """Execute the full feature engineering pipeline."""
    processed_dir = Path(config["data"]["processed_dir"])
    output_dir = processed_dir

    # Load a label variant to get texts and splits (u_zeros is always available)
    df = pd.read_parquet(processed_dir / "labels_u_zeros.parquet")
    print(f"Loaded {len(df):,} studies for feature engineering")

    # Step 1: Build TF-IDF features
    print("\n=== Building TF-IDF Features ===")
    build_tfidf(df, config, output_dir)

    # Step 2: Pre-cache RadBERT tokenizer
    print("\n=== Caching RadBERT Tokenizer ===")
    try:
        from transformers import AutoTokenizer

        tokenizer_name = config["radbert"]["model_name"]
        print(f"  Downloading/caching {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print(f"  Tokenizer vocab size: {tokenizer.vocab_size:,}")
        print(f"  Max model length: {tokenizer.model_max_length}")

        # Test tokenization
        sample_text = df.iloc[0]["report_text"]
        tokens = tokenizer(sample_text, max_length=512, truncation=True)
        print(f"  Sample tokenization: {len(tokens['input_ids'])} tokens")
    except Exception as e:
        print(f"  Warning: Could not cache RadBERT tokenizer: {e}")
        print("  RadBERT tokenization will happen on-the-fly during training.")

    # Step 3: Save label arrays for quick loading
    print("\n=== Pre-computing Label Arrays ===")
    for strategy in config["strategies"]:
        label_path = processed_dir / f"labels_{strategy}.parquet"
        if not label_path.exists():
            print(f"  Skipping {strategy} (not found)")
            continue

        df_strategy = pd.read_parquet(label_path)
        for split in ["train", "val", "test"]:
            labels, mask = prepare_labels_and_mask(df_strategy, split, PATHOLOGIES)
            np.save(output_dir / f"labels_{strategy}_{split}.npy", labels)
            np.save(output_dir / f"mask_{strategy}_{split}.npy", mask)
            n = labels.shape[0]
            n_masked = (mask == 0).sum()
            print(f"  {strategy}/{split}: {n:,} samples, {n_masked:,} masked labels")

    print("\nPhase 3 complete!")


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    run_feature_engineering(config)
