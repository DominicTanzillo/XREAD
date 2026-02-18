"""Phase 1: Data Processing Pipeline.

Loads CheXpert Plus CSV + CheXbert labels, merges, deduplicates to study level,
creates patient-level splits, and generates 4 label variant parquets.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
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


def load_chexpert_plus(csv_path: str) -> pd.DataFrame:
    """Load the CheXpert Plus CSV (223K rows)."""
    print(f"Loading CheXpert Plus CSV from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def load_chexbert_labels(json_path: str) -> pd.DataFrame:
    """Load CheXbert labels from JSONL (one JSON object per line)."""
    print(f"Loading CheXbert labels from {json_path}...")
    records = []
    with open(json_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="  Reading labels"):
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} label records")
    return df


def extract_study_key(path: str) -> str:
    """Extract patient/study identifier from image path.

    Example: 'CheXpert-v1.0/train/patient00001/study1/view1_frontal.jpg'
    Returns: 'patient00001/study1'
    """
    match = re.search(r"(patient\d+/study\d+)", str(path))
    if match:
        return match.group(1)
    return str(path)


def extract_patient_id(study_key: str) -> str:
    """Extract patient ID from study key.

    Example: 'patient00001/study1' -> 'patient00001'
    """
    return study_key.split("/")[0]


def map_kaggle_path(study_key: str, kaggle_base: str) -> str:
    """Map study key to CheXpert-small image directory path."""
    return f"{kaggle_base}/train/{study_key}"


def create_report_text(row: pd.Series) -> str:
    """Create unified report text: findings first, fallback to impression."""
    findings = str(row.get("section_findings", "")) if pd.notna(row.get("section_findings")) else ""
    impression = str(row.get("section_impression", "")) if pd.notna(row.get("section_impression")) else ""

    findings = findings.strip()
    impression = impression.strip()

    if findings and findings.lower() not in ("nan", "none", ""):
        return findings
    if impression and impression.lower() not in ("nan", "none", ""):
        return impression
    return ""


def normalize_text(text: str) -> str:
    """Normalize whitespace and clean text."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def compute_class_weights(labels_df: pd.DataFrame, pathologies: list[str]) -> dict:
    """Compute pos_weight for BCEWithLogitsLoss per pathology.

    pos_weight = num_negatives / num_positives
    """
    weights = {}
    for path in pathologies:
        if path not in labels_df.columns:
            weights[path] = 1.0
            continue
        col = labels_df[path]
        n_pos = (col == 1.0).sum()
        n_neg = (col == 0.0).sum()
        if n_pos > 0:
            weights[path] = n_neg / n_pos
        else:
            weights[path] = 1.0
    return weights


def make_label_variants(df: pd.DataFrame, pathologies: list[str]) -> dict[str, pd.DataFrame]:
    """Generate 4 label variant DataFrames from raw labels.

    - u_zeros: uncertain (-1) -> 0
    - u_ones: uncertain (-1) -> 1
    - u_ignore: uncertain (-1) -> NaN (masked from loss)
    - u_llm: placeholder, filled by Phase 2
    """
    variants = {}

    # U-Zeros: map -1 -> 0, null -> 0
    df_zeros = df.copy()
    for p in pathologies:
        if p in df_zeros.columns:
            df_zeros[p] = df_zeros[p].fillna(0.0)
            df_zeros.loc[df_zeros[p] == -1.0, p] = 0.0
    variants["u_zeros"] = df_zeros

    # U-Ones: map -1 -> 1, null -> 0
    df_ones = df.copy()
    for p in pathologies:
        if p in df_ones.columns:
            df_ones[p] = df_ones[p].fillna(0.0)
            df_ones.loc[df_ones[p] == -1.0, p] = 1.0
    variants["u_ones"] = df_ones

    # U-Ignore: map -1 -> NaN (masked), null -> 0
    df_ignore = df.copy()
    for p in pathologies:
        if p in df_ignore.columns:
            df_ignore[p] = df_ignore[p].fillna(0.0)
            df_ignore.loc[df_ignore[p] == -1.0, p] = np.nan
    variants["u_ignore"] = df_ignore

    # U-LLM: initially same as U-Ignore (Phase 2 fills in resolved values)
    df_llm = df.copy()
    for p in pathologies:
        if p in df_llm.columns:
            df_llm[p] = df_llm[p].fillna(0.0)
            # Keep -1 as-is for now — Phase 2 will resolve them
    variants["u_llm"] = df_llm

    return variants


def patient_level_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Split data at patient level to prevent data leakage.

    All studies for a given patient stay in the same split.
    """
    rng = np.random.RandomState(seed)

    patients = df["patient_id"].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_patients = set(patients[:n_test])
    val_patients = set(patients[n_test : n_test + n_val])
    train_patients = set(patients[n_test + n_val :])

    df = df.copy()
    df["split"] = "train"
    df.loc[df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["patient_id"].isin(test_patients), "split"] = "test"

    print(f"  Split sizes - Train: {(df['split'] == 'train').sum():,}, "
          f"Val: {(df['split'] == 'val').sum():,}, "
          f"Test: {(df['split'] == 'test').sum():,}")
    print(f"  Patient counts - Train: {len(train_patients):,}, "
          f"Val: {len(val_patients):,}, Test: {len(test_patients):,}")

    return df


def run_data_pipeline(config: dict):
    """Execute the full data processing pipeline."""
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load CheXpert Plus CSV
    csv_path = config["data"]["chexpert_plus_csv"]
    df_reports = load_chexpert_plus(csv_path)

    # Step 2: Load CheXbert labels
    labels_path = config["data"]["chexbert_labels_json"]
    df_labels = load_chexbert_labels(labels_path)

    # Step 3: Extract study keys for merging
    print("Extracting study keys...")
    if "path_to_image" in df_reports.columns:
        path_col = "path_to_image"
    else:
        # Try common alternatives
        path_cols = [c for c in df_reports.columns if "path" in c.lower()]
        path_col = path_cols[0] if path_cols else df_reports.columns[0]
        print(f"  Using column '{path_col}' as path column")

    df_reports["study_key"] = df_reports[path_col].apply(extract_study_key)

    # Labels may also have a path column
    label_path_cols = [c for c in df_labels.columns if "path" in c.lower() or c == path_col]
    if label_path_cols:
        label_path_col = label_path_cols[0]
    else:
        label_path_col = df_labels.columns[0]
    df_labels["study_key"] = df_labels[label_path_col].apply(extract_study_key)

    # Step 4: Merge on study_key
    print("Merging reports with labels...")
    # Keep only label columns + study_key from labels
    label_cols = [c for c in df_labels.columns if c in PATHOLOGIES or c == "study_key"]
    df_merged = df_reports.merge(df_labels[label_cols], on="study_key", how="inner")
    print(f"  Merged dataset: {len(df_merged):,} rows")

    # Step 5: Deduplicate to study level
    print("Deduplicating to study level...")
    df_studies = df_merged.drop_duplicates(subset=["study_key"]).reset_index(drop=True)
    print(f"  Unique studies: {len(df_studies):,}")

    # Step 6: Create unified report text
    print("Creating report text...")
    df_studies["report_text"] = df_studies.apply(create_report_text, axis=1)
    df_studies["report_text"] = df_studies["report_text"].apply(normalize_text)

    # Count empty reports
    n_empty = (df_studies["report_text"] == "").sum()
    print(f"  Empty reports after fallback: {n_empty:,}")
    if n_empty > 0:
        print("  Dropping empty reports...")
        df_studies = df_studies[df_studies["report_text"] != ""].reset_index(drop=True)
        print(f"  Remaining studies: {len(df_studies):,}")

    # Step 7: Map Kaggle paths
    kaggle_base = config["data"].get("kaggle_images_dir", "data/raw/CheXpert-v1.0-small")
    df_studies["kaggle_image_dir"] = df_studies["study_key"].apply(
        lambda x: map_kaggle_path(x, kaggle_base)
    )

    # Step 8: Extract patient IDs and split
    print("Creating patient-level splits...")
    df_studies["patient_id"] = df_studies["study_key"].apply(extract_patient_id)
    df_studies = patient_level_split(
        df_studies,
        val_size=config["processing"]["val_size"],
        test_size=config["processing"]["test_size"],
        seed=config["processing"]["random_seed"],
    )

    # Step 9: Treat unmentioned (null) labels as negative (0) — done inside make_label_variants

    # Step 10: Generate label variants
    print("Generating label variants...")
    variants = make_label_variants(df_studies, PATHOLOGIES)

    # Metadata columns to keep alongside labels
    meta_cols = ["study_key", "patient_id", "report_text", "kaggle_image_dir", "split"]

    for name, df_variant in variants.items():
        out_path = processed_dir / f"labels_{name}.parquet"
        cols_to_save = meta_cols + [p for p in PATHOLOGIES if p in df_variant.columns]
        df_variant[cols_to_save].to_parquet(out_path, index=False)
        print(f"  Saved {out_path} ({len(df_variant):,} rows)")

    # Step 11: Compute and save class weights (using u_zeros as reference)
    print("Computing class weights...")
    weights = compute_class_weights(variants["u_zeros"], PATHOLOGIES)
    weights_path = processed_dir / "class_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights, f, indent=2)
    print("  Class weights per pathology:")
    for p, w in weights.items():
        print(f"    {p}: {w:.2f}")

    # Summary statistics
    print("\n=== Data Pipeline Summary ===")
    print(f"Total studies: {len(df_studies):,}")
    print(f"Unique patients: {df_studies['patient_id'].nunique():,}")
    print(f"Label variants saved: {list(variants.keys())}")

    # Count uncertain labels
    n_uncertain = 0
    for p in PATHOLOGIES:
        if p in df_studies.columns:
            n_uncertain += (df_studies[p] == -1.0).sum()
    print(f"Total uncertain label instances: {n_uncertain:,}")

    print("\nPhase 1 complete!")
    return df_studies


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    run_data_pipeline(config)
