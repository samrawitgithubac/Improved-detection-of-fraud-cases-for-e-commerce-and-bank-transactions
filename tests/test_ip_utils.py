import pandas as pd

from src.data.ip_utils import attach_country_by_ip_range, ip_to_int


def test_ip_to_int_dotted_quad():
    assert ip_to_int("0.0.0.0") == 0
    assert ip_to_int("255.255.255.255") == 4294967295


def test_ip_to_int_numeric_strings():
    assert ip_to_int("123") == 123
    assert ip_to_int(123) == 123


def test_attach_country_by_ip_range_basic():
    fraud = pd.DataFrame({"ip_address": [1, 5, 10, 11]})
    ip_map = pd.DataFrame(
        {
            "lower_bound_ip_address": [0, 6],
            "upper_bound_ip_address": [5, 10],
            "country": ["A", "B"],
        }
    )
    out = attach_country_by_ip_range(fraud, ip_map, out_col="country")
    assert out["country"].tolist() == ["A", "A", "B", "Unknown"]


