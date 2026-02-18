"""XREAD: Radiology Report Pathology Classification with LLM-Based Uncertainty Resolution.

CLI entrypoint for all pipeline stages.
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/experiment.yaml") -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_data(args):
    """Run data processing pipeline (Phase 1)."""
    from scripts.make_dataset import run_data_pipeline

    config = load_config(args.config)
    run_data_pipeline(config)


def cmd_resolve(args):
    """Run LLM uncertainty resolution (Phase 2)."""
    from scripts.resolve_uncertainty import run_uncertainty_resolution

    config = load_config(args.config)
    run_uncertainty_resolution(
        config,
        sample_size=args.sample_size,
        skip_ollama=args.skip_ollama,
    )


def cmd_features(args):
    """Build features (Phase 3)."""
    from scripts.build_features import run_feature_engineering

    config = load_config(args.config)
    run_feature_engineering(config)


def cmd_train(args):
    """Train models (Phase 4)."""
    from scripts.train import run_training

    config = load_config(args.config)
    run_training(
        config,
        models=args.models,
        strategies=args.strategies,
    )


def cmd_evaluate(args):
    """Evaluate models (Phase 5)."""
    from scripts.evaluate import run_evaluation

    config = load_config(args.config)
    run_evaluation(config)


def cmd_app(args):
    """Launch Gradio app (Phase 6)."""
    from scripts.app import launch_app

    config = load_config(args.config)
    launch_app(config, share=args.share)


def main():
    parser = argparse.ArgumentParser(
        prog="xread",
        description="XREAD: Radiology Report Pathology Classification",
    )
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment config YAML",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # Data pipeline
    sub_data = subparsers.add_parser("data", help="Run data processing pipeline")
    sub_data.set_defaults(func=cmd_data)

    # Uncertainty resolution
    sub_resolve = subparsers.add_parser("resolve", help="Run LLM uncertainty resolution")
    sub_resolve.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of reports to process with Ollama (None=all remaining)",
    )
    sub_resolve.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama pass, only run regex pre-filter",
    )
    sub_resolve.set_defaults(func=cmd_resolve)

    # Feature engineering
    sub_features = subparsers.add_parser("features", help="Build features")
    sub_features.set_defaults(func=cmd_features)

    # Training
    sub_train = subparsers.add_parser("train", help="Train models")
    sub_train.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["naive", "classical", "radbert"],
        help="Which models to train (default: all)",
    )
    sub_train.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=["u_zeros", "u_ones", "u_ignore", "u_llm"],
        help="Which uncertainty strategies (default: all)",
    )
    sub_train.set_defaults(func=cmd_train)

    # Evaluation
    sub_eval = subparsers.add_parser("evaluate", help="Evaluate models")
    sub_eval.set_defaults(func=cmd_evaluate)

    # App
    sub_app = subparsers.add_parser("app", help="Launch Gradio app")
    sub_app.add_argument("--share", action="store_true", help="Create public Gradio link")
    sub_app.set_defaults(func=cmd_app)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
