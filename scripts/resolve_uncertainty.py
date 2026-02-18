"""Phase 2: LLM Uncertainty Resolution (Novel Contribution).

Two-pass approach:
  Pass 1: Regex pre-filter resolves ~50-60% of uncertain labels.
  Pass 2: Ollama LLM classifies remaining ambiguous reports.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
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

# Resolution types
HEDGED_POSITIVE = "HEDGED_POSITIVE"
LOW_CONFIDENCE_NEGATIVE = "LOW_CONFIDENCE_NEGATIVE"
DIFFERENTIAL = "DIFFERENTIAL"
TRUE_AMBIGUITY = "TRUE_AMBIGUITY"

# Regex patterns for pre-filter (Pass 1)
HEDGED_POSITIVE_PATTERNS = [
    r"cannot\s+exclude",
    r"concerning\s+for",
    r"suspicious\s+for",
    r"may\s+represent",
    r"possible\s+",
    r"could\s+represent",
    r"suggestive\s+of",
    r"cannot\s+be\s+excluded",
    r"might\s+be",
    r"potentially",
]

LOW_CONFIDENCE_NEGATIVE_PATTERNS = [
    r"no\s+definite",
    r"unlikely",
    r"low\s+probability",
    r"no\s+evidence\s+of",
    r"no\s+convincing",
    r"no\s+significant",
    r"no\s+definitive",
    r"probably\s+normal",
    r"no\s+acute",
    r"no\s+obvious",
]

DIFFERENTIAL_PATTERNS = [
    r"\bvs\.\s",
    r"\bversus\b",
    r"differential\s+includes",
    r"differential\s+diagnosis",
    r"\bor\s+possibly\b",
]


OLLAMA_PROMPT_TEMPLATE = """You are a radiology report classifier. Given a radiology report excerpt and a specific pathology finding that was labeled as "uncertain" by an NLP system, classify the TYPE of uncertainty.

Report excerpt: "{report_text}"

Uncertain pathology: "{pathology}"

Classify this uncertainty into exactly ONE category:
A) HEDGED_POSITIVE - The radiologist believes the condition is likely present but uses hedging language (e.g., "cannot exclude", "suspicious for", "may represent")
B) TRUE_AMBIGUITY - The report is genuinely ambiguous and the presence/absence cannot be determined from the text alone
C) DIFFERENTIAL - The condition is mentioned as part of a differential diagnosis alongside other possibilities
D) LOW_CONFIDENCE_NEGATIVE - The radiologist leans toward the condition being absent but cannot fully rule it out (e.g., "no definite", "unlikely", "low probability")

Respond with ONLY the single letter (A, B, C, or D)."""


def regex_classify(report_text: str, pathology: str) -> str | None:
    """Classify uncertainty type using regex patterns.

    Returns resolution type string or None if no pattern matches.
    """
    text_lower = report_text.lower()

    # Check for hedged positive patterns near the pathology mention
    pathology_lower = pathology.lower()

    for pattern in HEDGED_POSITIVE_PATTERNS:
        if re.search(pattern, text_lower):
            # Check if pattern is near the pathology mention
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 100)
                end = min(len(text_lower), match.end() + 100)
                context = text_lower[start:end]
                if pathology_lower in context or _fuzzy_pathology_match(context, pathology_lower):
                    return HEDGED_POSITIVE

    for pattern in LOW_CONFIDENCE_NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower):
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 100)
                end = min(len(text_lower), match.end() + 100)
                context = text_lower[start:end]
                if pathology_lower in context or _fuzzy_pathology_match(context, pathology_lower):
                    return LOW_CONFIDENCE_NEGATIVE

    for pattern in DIFFERENTIAL_PATTERNS:
        if re.search(pattern, text_lower):
            for match in re.finditer(pattern, text_lower):
                start = max(0, match.start() - 100)
                end = min(len(text_lower), match.end() + 100)
                context = text_lower[start:end]
                if pathology_lower in context or _fuzzy_pathology_match(context, pathology_lower):
                    return DIFFERENTIAL

    return None


def _fuzzy_pathology_match(context: str, pathology: str) -> bool:
    """Check if pathology or related terms appear in context."""
    # Map pathology names to common report terms
    aliases = {
        "enlarged cardiomediastinum": ["cardiomediastinum", "mediastinum", "mediastinal"],
        "cardiomegaly": ["cardiomegaly", "cardiac", "heart size", "heart is enlarged"],
        "lung opacity": ["opacity", "opacities", "opacification"],
        "lung lesion": ["lesion", "mass", "nodule", "nodular"],
        "edema": ["edema", "pulmonary edema", "vascular congestion", "fluid"],
        "consolidation": ["consolidation", "consolidated"],
        "pneumonia": ["pneumonia", "infection", "infectious"],
        "atelectasis": ["atelectasis", "atelectatic"],
        "pneumothorax": ["pneumothorax"],
        "pleural effusion": ["pleural effusion", "effusion", "pleural fluid"],
        "pleural other": ["pleural", "pleurisy", "pleural thickening"],
        "fracture": ["fracture", "fractured", "broken"],
        "support devices": ["support device", "line", "tube", "catheter", "pacemaker"],
        "no finding": ["no finding", "normal", "unremarkable"],
    }
    terms = aliases.get(pathology, [pathology])
    return any(term in context for term in terms)


def ollama_classify(
    report_text: str,
    pathology: str,
    base_url: str,
    model: str,
    temperature: float = 0,
    num_predict: int = 5,
) -> str:
    """Send report to Ollama LLM for uncertainty classification."""
    prompt = OLLAMA_PROMPT_TEMPLATE.format(
        report_text=report_text[:1500],  # Truncate very long reports
        pathology=pathology,
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    try:
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        answer = result.get("response", "").strip().upper()

        # Parse single letter response
        if "A" in answer:
            return HEDGED_POSITIVE
        elif "D" in answer:
            return LOW_CONFIDENCE_NEGATIVE
        elif "C" in answer:
            return DIFFERENTIAL
        elif "B" in answer:
            return TRUE_AMBIGUITY
        else:
            return TRUE_AMBIGUITY  # Default fallback
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  Ollama error: {e}")
        return TRUE_AMBIGUITY  # Fallback on error


def resolution_to_label(resolution_type: str) -> float:
    """Map resolution type to label value."""
    mapping = {
        HEDGED_POSITIVE: 1.0,
        LOW_CONFIDENCE_NEGATIVE: 0.0,
        DIFFERENTIAL: 1.0,
        TRUE_AMBIGUITY: float("nan"),  # Mask from loss
    }
    return mapping.get(resolution_type, float("nan"))


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load existing checkpoint resolutions."""
    resolutions = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    key = f"{record['study_key']}|{record['pathology']}"
                    resolutions[key] = record
        print(f"  Loaded {len(resolutions)} checkpoint resolutions")
    return resolutions


def save_checkpoint(checkpoint_path: Path, resolutions: list[dict]):
    """Append resolutions to checkpoint file."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for r in resolutions:
            f.write(json.dumps(r) + "\n")


def run_uncertainty_resolution(
    config: dict,
    sample_size: int | None = None,
    skip_ollama: bool = False,
):
    """Execute the full uncertainty resolution pipeline."""
    processed_dir = Path(config["data"]["processed_dir"])
    ollama_config = config["ollama"]

    # Load the U-LLM label variant (has -1.0 for uncertain)
    llm_path = processed_dir / "labels_u_llm.parquet"
    if not llm_path.exists():
        print("Error: labels_u_llm.parquet not found. Run 'python main.py data' first.")
        return

    df = pd.read_parquet(llm_path)
    print(f"Loaded {len(df):,} studies")

    # Find all uncertain instances
    uncertain_instances = []
    for _, row in df.iterrows():
        for p in PATHOLOGIES:
            if p in df.columns and row[p] == -1.0:
                uncertain_instances.append({
                    "study_key": row["study_key"],
                    "report_text": row["report_text"],
                    "pathology": p,
                })

    print(f"Found {len(uncertain_instances):,} uncertain label instances "
          f"across {len(set(u['study_key'] for u in uncertain_instances)):,} reports")

    if not uncertain_instances:
        print("No uncertain labels found. Nothing to resolve.")
        return

    # === Pass 1: Regex pre-filter ===
    print("\n=== Pass 1: Regex Pre-filter ===")
    resolutions = {}
    regex_resolved = 0

    for inst in tqdm(uncertain_instances, desc="Regex classification"):
        key = f"{inst['study_key']}|{inst['pathology']}"
        result = regex_classify(inst["report_text"], inst["pathology"])
        if result is not None:
            resolutions[key] = {
                "study_key": inst["study_key"],
                "pathology": inst["pathology"],
                "resolution": result,
                "method": "regex",
            }
            regex_resolved += 1

    print(f"  Regex resolved: {regex_resolved:,} / {len(uncertain_instances):,} "
          f"({100 * regex_resolved / len(uncertain_instances):.1f}%)")

    # Count by type
    type_counts = {}
    for r in resolutions.values():
        t = r["resolution"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c:,}")

    # === Pass 2: Ollama LLM ===
    remaining = [
        inst for inst in uncertain_instances
        if f"{inst['study_key']}|{inst['pathology']}" not in resolutions
    ]
    print(f"\n=== Pass 2: Ollama LLM ({len(remaining):,} remaining) ===")

    if skip_ollama:
        print("  Skipping Ollama pass (--skip-ollama flag)")
        # Assign remaining as TRUE_AMBIGUITY
        for inst in remaining:
            key = f"{inst['study_key']}|{inst['pathology']}"
            resolutions[key] = {
                "study_key": inst["study_key"],
                "pathology": inst["pathology"],
                "resolution": TRUE_AMBIGUITY,
                "method": "skip_default",
            }
    else:
        # Setup checkpoint
        checkpoint_dir = Path(ollama_config["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "ollama_resolutions.jsonl"

        # Load existing checkpoint
        existing = load_checkpoint(checkpoint_path)
        for key, record in existing.items():
            if key not in resolutions:
                resolutions[key] = record

        # Filter remaining to only those not yet resolved
        remaining = [
            inst for inst in remaining
            if f"{inst['study_key']}|{inst['pathology']}" not in resolutions
        ]
        print(f"  After checkpoint: {len(remaining):,} still unresolved")

        if sample_size is not None and len(remaining) > sample_size:
            print(f"  Sampling {sample_size:,} from {len(remaining):,}")
            rng = np.random.RandomState(42)
            indices = rng.choice(len(remaining), size=sample_size, replace=False)
            sampled = [remaining[i] for i in indices]
            # Mark unsampled as TRUE_AMBIGUITY
            sampled_keys = {f"{inst['study_key']}|{inst['pathology']}" for inst in sampled}
            for inst in remaining:
                key = f"{inst['study_key']}|{inst['pathology']}"
                if key not in sampled_keys:
                    resolutions[key] = {
                        "study_key": inst["study_key"],
                        "pathology": inst["pathology"],
                        "resolution": TRUE_AMBIGUITY,
                        "method": "unsampled_default",
                    }
            remaining = sampled

        # Process with Ollama
        batch = []
        for i, inst in enumerate(tqdm(remaining, desc="Ollama classification")):
            key = f"{inst['study_key']}|{inst['pathology']}"
            result = ollama_classify(
                inst["report_text"],
                inst["pathology"],
                ollama_config["base_url"],
                ollama_config["model"],
                ollama_config["temperature"],
                ollama_config["num_predict"],
            )
            record = {
                "study_key": inst["study_key"],
                "pathology": inst["pathology"],
                "resolution": result,
                "method": "ollama",
            }
            resolutions[key] = record
            batch.append(record)

            # Checkpoint periodically
            if len(batch) >= ollama_config["checkpoint_interval"]:
                save_checkpoint(checkpoint_path, batch)
                batch = []

        # Save remaining batch
        if batch:
            save_checkpoint(checkpoint_path, batch)

    # === Apply resolutions to labels ===
    print("\n=== Applying resolutions ===")
    for idx, row in df.iterrows():
        for p in PATHOLOGIES:
            if p in df.columns and row[p] == -1.0:
                key = f"{row['study_key']}|{p}"
                if key in resolutions:
                    df.at[idx, p] = resolution_to_label(resolutions[key]["resolution"])
                else:
                    df.at[idx, p] = float("nan")  # Unresolved -> mask

    # Verify no -1.0 remaining
    n_remaining = 0
    for p in PATHOLOGIES:
        if p in df.columns:
            n_remaining += (df[p] == -1.0).sum()
    print(f"  Remaining -1.0 labels: {n_remaining}")
    assert n_remaining == 0, "All uncertain labels should be resolved!"

    # Save resolved labels
    df.to_parquet(llm_path, index=False)
    print(f"  Saved resolved labels to {llm_path}")

    # Save resolution details
    resolution_list = list(resolutions.values())
    resolutions_path = processed_dir / "uncertainty_resolutions.json"
    with open(resolutions_path, "w") as f:
        json.dump(resolution_list, f, indent=2)

    # Compute and save stats
    stats = {
        "total_uncertain": len(uncertain_instances),
        "regex_resolved": sum(1 for r in resolution_list if r["method"] == "regex"),
        "ollama_resolved": sum(1 for r in resolution_list if r["method"] == "ollama"),
        "by_type": {},
    }
    for r in resolution_list:
        t = r["resolution"]
        stats["by_type"][t] = stats["by_type"].get(t, 0) + 1

    stats_path = processed_dir / "uncertainty_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Resolution Summary ===")
    print(f"  Total uncertain instances: {stats['total_uncertain']:,}")
    print(f"  Regex resolved: {stats['regex_resolved']:,}")
    print(f"  Ollama resolved: {stats['ollama_resolved']:,}")
    print(f"  By type:")
    for t, c in sorted(stats["by_type"].items()):
        print(f"    {t}: {c:,}")

    print("\nPhase 2 complete!")


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    run_uncertainty_resolution(config)
