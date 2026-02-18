"""Phase 6: Gradio Deployment App.

Three-tab clinical workflow UI:
  Tab 1 - Analyze Report: Paste report text, get pathology predictions
  Tab 2 - Compare Models: Interactive 4×3 results table
  Tab 3 - About: Methodology, limitations, links
"""

import json
import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

from scripts.model import PATHOLOGIES, PerPathologyClassifier, RadBERTClassifier


EXAMPLE_REPORTS = [
    {
        "name": "Normal chest X-ray",
        "text": "The lungs are clear bilaterally. No pleural effusion or pneumothorax. "
                "The cardiac silhouette is normal in size. The mediastinum is unremarkable. "
                "No acute osseous abnormalities.",
    },
    {
        "name": "Pneumonia with effusion",
        "text": "There is a left lower lobe consolidation with air bronchograms, consistent "
                "with pneumonia. Small left pleural effusion is noted. The right lung is clear. "
                "Heart size is normal. No pneumothorax.",
    },
    {
        "name": "Cardiomegaly with edema",
        "text": "The cardiac silhouette is significantly enlarged. There is pulmonary vascular "
                "congestion and bilateral pleural effusions, right greater than left, consistent "
                "with congestive heart failure. Bilateral perihilar haziness suggestive of "
                "pulmonary edema. No pneumothorax.",
    },
    {
        "name": "Uncertain finding",
        "text": "Cannot exclude a small right apical pneumothorax. There is a questionable "
                "opacity in the left lower lobe which may represent atelectasis versus early "
                "consolidation. Clinical correlation recommended. Heart size is upper limits "
                "of normal. Support devices are in satisfactory position.",
    },
]


def load_models(models_dir: Path, config: dict) -> dict:
    """Load all available trained models."""
    loaded = {}

    strategies = config["strategies"]
    model_types = config["models"]

    for strategy in strategies:
        for model_type in model_types:
            key = f"{model_type}_{strategy}"

            if model_type == "classical":
                path = models_dir / f"classical_{strategy}.pkl"
                if path.exists():
                    loaded[key] = {
                        "type": "classical",
                        "model": PerPathologyClassifier.load(str(path)),
                    }

            elif model_type == "radbert":
                path = models_dir / f"radbert_{strategy}.pt"
                if path.exists():
                    device = torch.device("cpu")  # CPU for deployment
                    model = RadBERTClassifier(
                        model_name=config["radbert"]["model_name"],
                        num_labels=len(PATHOLOGIES),
                        dropout=config["radbert"]["dropout"],
                    )
                    model.load_state_dict(
                        torch.load(path, map_location=device, weights_only=True)
                    )
                    model.eval()
                    loaded[key] = {
                        "type": "radbert",
                        "model": model,
                    }

    return loaded


def predict_single_report(
    report_text: str,
    model_key: str,
    loaded_models: dict,
    tfidf_vectorizer,
    tokenizer,
    max_length: int = 512,
) -> list[tuple[str, float]]:
    """Get predictions for a single report."""
    if model_key not in loaded_models:
        return [(p, 0.5) for p in PATHOLOGIES]

    model_info = loaded_models[model_key]

    if model_info["type"] == "classical":
        X = tfidf_vectorizer.transform([report_text])
        probs = model_info["model"].predict_proba(X)[0]

    elif model_info["type"] == "radbert":
        encoding = tokenizer(
            report_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model_info["model"](
                encoding["input_ids"],
                encoding["attention_mask"],
            )
            probs = torch.sigmoid(logits).squeeze().numpy()

    else:
        probs = np.full(len(PATHOLOGIES), 0.5)

    return [(PATHOLOGIES[i], float(probs[i])) for i in range(len(PATHOLOGIES))]


def format_predictions_html(predictions: list[tuple[str, float]]) -> str:
    """Format predictions as color-coded HTML bars."""
    html = '<div style="font-family: monospace; max-width: 600px;">'

    for pathology, prob in predictions:
        # Color: green (<0.3), yellow (0.3-0.7), red (>0.7)
        if prob < 0.3:
            color = "#4CAF50"
            label = "Negative"
        elif prob < 0.7:
            color = "#FF9800"
            label = "Borderline"
        else:
            color = "#F44336"
            label = "Positive"

        bar_width = max(5, int(prob * 100))

        html += f'''
        <div style="margin: 4px 0; display: flex; align-items: center;">
            <span style="width: 220px; font-size: 13px;">{pathology}</span>
            <div style="flex: 1; background: #eee; border-radius: 4px; height: 20px; margin: 0 8px;">
                <div style="width: {bar_width}%; background: {color}; height: 100%;
                            border-radius: 4px; min-width: 5%;"></div>
            </div>
            <span style="width: 120px; font-size: 12px; color: {color};">
                {prob:.3f} ({label})
            </span>
        </div>'''

    html += "</div>"
    return html


def create_results_table(output_dir: Path) -> str:
    """Load and format the 4×3 results table."""
    table_path = output_dir / "results_table.csv"
    if not table_path.exists():
        return "Results table not found. Run evaluation first."

    df = pd.read_csv(table_path, index_col=0)
    return df.to_html(float_format="%.4f", classes="results-table")


def launch_app(config: dict, share: bool = False):
    """Launch the Gradio application."""
    models_dir = Path("models")
    output_dir = Path(config["data"]["outputs_dir"])

    # Load models
    print("Loading models...")
    loaded_models = load_models(models_dir, config)
    available_models = list(loaded_models.keys())
    print(f"  Loaded {len(available_models)} models: {available_models}")

    # Load TF-IDF vectorizer
    tfidf_vectorizer = None
    vec_path = Path(config["tfidf"]["vectorizer_path"])
    if vec_path.exists():
        with open(vec_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        print("  Loaded TF-IDF vectorizer")

    # Load RadBERT tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config["radbert"]["model_name"])
        print("  Loaded RadBERT tokenizer")
    except Exception as e:
        print(f"  Warning: Could not load RadBERT tokenizer: {e}")

    # === Tab 1: Analyze Report ===
    def analyze_report(report_text, model_choice):
        if not report_text.strip():
            return "Please enter a radiology report."

        predictions = predict_single_report(
            report_text, model_choice,
            loaded_models, tfidf_vectorizer, tokenizer,
            max_length=config["radbert"]["max_length"],
        )
        return format_predictions_html(predictions)

    def load_example(example_name):
        for ex in EXAMPLE_REPORTS:
            if ex["name"] == example_name:
                return ex["text"]
        return ""

    # === Tab 2: Compare Models ===
    def compare_on_report(report_text):
        if not report_text.strip():
            return "Please enter a report first."

        results = {}
        for key in available_models:
            preds = predict_single_report(
                report_text, key,
                loaded_models, tfidf_vectorizer, tokenizer,
                max_length=config["radbert"]["max_length"],
            )
            results[key] = {p: prob for p, prob in preds}

        if not results:
            return "No models available."

        # Format as comparison table
        df = pd.DataFrame(results).T
        df.index.name = "Model"
        return df.to_html(float_format="%.3f")

    # Build Gradio interface
    with gr.Blocks(title=config["app"]["title"], theme=gr.themes.Soft()) as app:
        gr.Markdown(f"# {config['app']['title']}")
        gr.Markdown("Classify 14 pathology findings from radiology report text using "
                     "multiple models and uncertainty resolution strategies.")

        with gr.Tabs():
            # Tab 1: Analyze Report
            with gr.Tab("Analyze Report"):
                with gr.Row():
                    with gr.Column(scale=2):
                        report_input = gr.Textbox(
                            label="Radiology Report",
                            placeholder="Paste a radiology report here...",
                            lines=8,
                        )
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=available_models if available_models else ["No models loaded"],
                                value=available_models[0] if available_models else "No models loaded",
                                label="Model / Strategy",
                            )
                            example_dropdown = gr.Dropdown(
                                choices=[ex["name"] for ex in EXAMPLE_REPORTS],
                                label="Load Example",
                            )
                        analyze_btn = gr.Button("Analyze", variant="primary")

                    with gr.Column(scale=3):
                        output_html = gr.HTML(label="Predictions")

                analyze_btn.click(
                    fn=analyze_report,
                    inputs=[report_input, model_dropdown],
                    outputs=output_html,
                )
                example_dropdown.change(
                    fn=load_example,
                    inputs=example_dropdown,
                    outputs=report_input,
                )

            # Tab 2: Compare Models
            with gr.Tab("Compare Models"):
                gr.Markdown("### Experiment Results (4x3 Matrix)")
                results_html = gr.HTML(value=create_results_table(output_dir))

                gr.Markdown("### Side-by-side Predictions")
                compare_input = gr.Textbox(
                    label="Report Text",
                    placeholder="Enter a report to compare predictions across all models...",
                    lines=5,
                )
                compare_btn = gr.Button("Compare All Models")
                compare_output = gr.HTML()

                compare_btn.click(
                    fn=compare_on_report,
                    inputs=compare_input,
                    outputs=compare_output,
                )

            # Tab 3: About
            with gr.Tab("About"):
                gr.Markdown("""
### Methodology

**XREAD** uses a novel approach to handle uncertain labels in the CheXpert dataset:

1. **Data**: CheXpert Plus (223K radiology reports) + CheXbert pathology labels
2. **Uncertainty Resolution**: Instead of blanket strategies (U-Zeros, U-Ones, U-Ignore),
   we use a two-pass system:
   - **Pass 1**: Regex patterns classify common hedging language
   - **Pass 2**: Local LLM (Ollama) reads remaining ambiguous reports
3. **Models**: Majority baseline, TF-IDF + Logistic Regression, Fine-tuned RadBERT
4. **Evaluation**: Per-pathology AUROC across 4×3 experiment matrix

### Model Card

- **RadBERT**: Stanford AIMI's radiology-specialized BERT model (110M params)
- **Training**: 5 epochs, batch size 32, fp16, early stopping on val macro-AUROC
- **14 Pathologies**: No Finding, Cardiomegaly, Edema, Consolidation, Atelectasis,
  Pleural Effusion, Pneumonia, Pneumothorax, and 6 more

### Limitations

- Text-only classification (no image analysis)
- Trained on CheXpert (US academic center) — may not generalize to other populations
- Uncertainty resolution is approximate — true clinical uncertainty requires expert review
- Not validated for clinical decision-making

### Links

- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [RadBERT](https://huggingface.co/StanfordAIMI/RadBERT)
- [GitHub Repository](https://github.com/DominicTanzillo/XREAD)
                """)

    print(f"\nLaunching Gradio app on port {config['app']['server_port']}...")
    app.launch(
        share=share,
        server_port=config["app"]["server_port"],
    )


if __name__ == "__main__":
    import yaml

    with open("configs/experiment.yaml") as f:
        config = yaml.safe_load(f)
    launch_app(config)
