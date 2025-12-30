from __future__ import annotations

import argparse
from pathlib import Path

from src.modeling.task2_train import Task2Paths, train_and_evaluate_task2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2: model training and evaluation.")
    p.add_argument("--dataset", required=True, choices=["fraud", "creditcard", "all"])
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional: use a fraction of the data for faster experimentation (e.g. 0.2).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = Task2Paths(
        raw_dir=Path(args.raw_dir),
        reports_dir=Path(args.reports_dir),
        models_dir=Path(args.models_dir),
    )

    if args.dataset in ("fraud", "all"):
        res = train_and_evaluate_task2(
            "fraud",
            paths,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_splits=args.cv_splits,
            sample_frac=args.sample_frac,
        )
        print("fraud Task2 complete. Wrote:", paths.reports_dir / "task2_fraud_results.json")
        print("Models:", list(res["models"].keys()))

    if args.dataset in ("creditcard", "all"):
        res = train_and_evaluate_task2(
            "creditcard",
            paths,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_splits=args.cv_splits,
            sample_frac=args.sample_frac,
        )
        print("creditcard Task2 complete. Wrote:", paths.reports_dir / "task2_creditcard_results.json")
        print("Models:", list(res["models"].keys()))


if __name__ == "__main__":
    main()


