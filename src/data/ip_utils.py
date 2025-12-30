from __future__ import annotations

import ipaddress
from typing import Union

import numpy as np
import pandas as pd


def ip_to_int(ip: Union[str, int, float]) -> int | None:
    """
    Convert an IP address to an integer.

    Supports:
    - dotted-quad strings: "192.168.0.1"
    - integer-like values (as int/float/str)

    Returns None if the value can't be parsed.
    """
    if ip is None:
        return None

    # pandas missing values
    if isinstance(ip, float) and np.isnan(ip):
        return None

    if isinstance(ip, (int, np.integer)):
        if ip < 0:
            return None
        return int(ip)

    if isinstance(ip, float):
        if ip < 0:
            return None
        # treat 1.0 as 1
        return int(ip)

    s = str(ip).strip()
    if not s:
        return None

    # integer string
    if s.isdigit():
        try:
            v = int(s)
            return v if v >= 0 else None
        except ValueError:
            return None

    try:
        return int(ipaddress.ip_address(s))
    except ValueError:
        return None


def attach_country_by_ip_range(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    *,
    fraud_ip_col: str = "ip_address",
    lower_col: str = "lower_bound_ip_address",
    upper_col: str = "upper_bound_ip_address",
    country_col: str = "country",
    out_col: str = "country",
) -> pd.DataFrame:
    """
    Range-based join of fraud transactions to IP→country mapping.

    Uses a fast strategy:
    - convert IP to int
    - merge_asof on lower_bound (direction='backward')
    - validate ip_int <= upper_bound, else set Unknown
    """
    df = fraud_df.copy()
    ip_map = ip_country_df.copy()

    df["_ip_int"] = df[fraud_ip_col].map(ip_to_int)
    ip_map["_lower"] = pd.to_numeric(ip_map[lower_col], errors="coerce")
    ip_map["_upper"] = pd.to_numeric(ip_map[upper_col], errors="coerce")

    ip_map = ip_map.dropna(subset=["_lower", "_upper", country_col]).sort_values("_lower")
    ip_map["_lower"] = ip_map["_lower"].astype("int64")
    ip_map["_upper"] = ip_map["_upper"].astype("int64")

    # merge_asof requires non-null, sorted keys
    df_valid = df[df["_ip_int"].notna()].copy()
    df_valid["_ip_int"] = df_valid["_ip_int"].astype("int64")
    df_valid = df_valid.sort_values("_ip_int")
    df_invalid = df[df["_ip_int"].isna()].copy()

    merged_valid = pd.merge_asof(
        df_valid,
        ip_map[["_lower", "_upper", country_col]],
        left_on="_ip_int",
        right_on="_lower",
        direction="backward",
    )

    valid = merged_valid["_upper"].notna() & (merged_valid["_ip_int"] <= merged_valid["_upper"])
    merged_valid[out_col] = np.where(valid, merged_valid[country_col], "Unknown")

    df_invalid[out_col] = "Unknown"

    merged = pd.concat([merged_valid, df_invalid], axis=0, ignore_index=True)

    cols_to_drop = ["_ip_int", "_lower", "_upper"]
    if country_col != out_col:
        cols_to_drop.append(country_col)

    merged = merged.drop(columns=[c for c in cols_to_drop if c in merged.columns])
    return merged


