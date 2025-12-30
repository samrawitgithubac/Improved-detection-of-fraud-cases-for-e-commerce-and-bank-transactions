from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.task3_shap import Task3Paths, explain_task3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: SHAP explainability.")
    p.add_argument("--dataset", required=True, choices=["fraud", "creditcard"])
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--model-name", default=None, help="Optional: override model key (e.g. random_forest)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--background-size", type=int, default=200)
    p.add_argument("--explain-size", type=int, default=2000)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = Task3Paths(
        raw_dir=Path(args.raw_dir),
        reports_dir=Path(args.reports_dir),
        models_dir=Path(args.models_dir),
    )

    res = explain_task3(
        dataset=args.dataset,
        paths=paths,
        model_name=args.model_name,
        test_size=args.test_size,
        random_state=args.random_state,
        background_size=args.background_size,
        explain_size=args.explain_size,
        threshold=args.threshold,
    )

    print("Dataset:", res["dataset"])
    print("Model:", res["model_name"])
    print("Model path:", res["model_path"])
    print("Example indices (in explained sample):", res["examples"])
    print("Note: SHAP plots are best viewed from the notebook: notebooks/shap-explainability.ipynb")


if __name__ == "__main__":
    main()


