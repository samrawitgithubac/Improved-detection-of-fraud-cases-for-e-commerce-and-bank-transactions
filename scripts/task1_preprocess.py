from __future__ import annotations

import argparse
from pathlib import Path

from src.data.creditcard_features import CreditcardTask1Paths, run_task1_creditcard
from src.data.fraud_features import FraudTask1Paths, run_task1_fraud


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task-1 preprocessing for fraud datasets.")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["fraud", "creditcard", "all"],
        help="Which dataset to preprocess.",
    )
    p.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing raw CSV files.",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory to write processed artifacts.",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.dataset in ("fraud", "all"):
        meta = run_task1_fraud(
            FraudTask1Paths(
                raw_fraud_csv=raw_dir / "Fraud_Data.csv",
                raw_ip_country_csv=raw_dir / "IpAddress_to_Country.csv",
                out_dir=out_dir,
            ),
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print("Fraud_Data task1 complete. Metadata:", meta)

    if args.dataset in ("creditcard", "all"):
        meta = run_task1_creditcard(
            CreditcardTask1Paths(raw_creditcard_csv=raw_dir / "creditcard.csv", out_dir=out_dir),
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print("creditcard task1 complete. Metadata:", meta)


if __name__ == "__main__":
    main()


