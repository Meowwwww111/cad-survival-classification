"""Main entry point: python main.py [train|predict] [options]."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining, predict

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    """Load experiment settings. Relative dataset/output paths use the project root."""
    with Path(path or ROOT / "src/config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    for key in ("file_path", "output_dir", "model_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else ROOT / value)
    if config["cv"] < 2 or config["holdout_folds"] < 2:
        raise ValueError("cv and holdout_folds must be at least 2")
    grid = config["threshold_grid"]
    if not (0 < grid["start"] < grid["stop"] < 1 and grid["count"] >= 2):
        raise ValueError("Threshold grid must contain at least two values within (0, 1)")
    if config["bootstrap_repetitions"] < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    if config["models"]["gradient_boosting"].get("early_stopping", False) is not False:
        raise ValueError("Keep early_stopping false to avoid an internal ungrouped validation split")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["train", "predict"], default="train")
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--data", help="CSV path; required for prediction")
    parser.add_argument("--output", help="Report directory, or output CSV for prediction")
    parser.add_argument("--model-dir", help="Directory for the saved training artifact")
    parser.add_argument("--model", help="Trusted joblib model file for prediction")
    parser.add_argument("--jobs", type=int, help="Cross-validation worker count")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "predict":
        if not args.data:
            parser.error("predict requires --data")
        args.model = args.model or str(Path(config["model_dir"]) / "best_model.joblib")
        args.output = args.output or str(Path(config["model_dir"]) / "predictions.csv")
        predict(args)
        return
    args.data = args.data or config["file_path"]
    args.output = args.output or config["output_dir"]
    args.model_dir = args.model_dir or config["model_dir"]
    args.jobs = args.jobs if args.jobs is not None else config["n_jobs"]
    data_preparation = DataPreparation(config)
    ModelTraining(config, data_preparation).run(args)


if __name__ == "__main__":
    main()
