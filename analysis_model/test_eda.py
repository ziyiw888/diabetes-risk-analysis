"""
Ziyi Wang
CSE 163 AC
Perform testing on EDA modules.
"""
import pandas as pd
import eda_utils as eda


def test_missing_table_schema() -> None:
    """
    missing_table returns 3 cols, row per original column.
    """
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [1, 2, 3],
        "c": [None, None, 5],
    })
    mt = eda.missing_table(df)

    assert set(["column", "n_missing", "pct_missing"]).issubset(mt.columns)
    assert len(mt) == df.shape[1]

    n_miss_a = mt.loc[mt["column"] == "a", "n_missing"].iloc[0]
    n_miss_b = mt.loc[mt["column"] == "b", "n_missing"].iloc[0]
    n_miss_c = mt.loc[mt["column"] == "c", "n_missing"].iloc[0]
    assert n_miss_a == 1
    assert n_miss_b == 0
    assert n_miss_c == 2

    assert (mt["pct_missing"] >= 0).all() and (mt["pct_missing"] <= 100).all()


def test_numeric_seven_number() -> None:
    """
    numeric_seven_number returns 7 stats for numeric cols only.
    """
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "z": ["a", "b", "c", "d", "e"],
    })
    out = eda.numeric_seven_number(df, ["x", "y", "z"])

    assert set(out.index.tolist()) == {"x", "y"}

    expected = ["mean", "std", "min", "q1", "median", "q3", "max"]
    for c in expected:
        assert c in out.columns

    assert out.loc["x", "min"] == 1
    assert out.loc["x", "max"] == 5
    assert out.loc["y", "median"] == 30


def main() -> None:
    print("[test_eda] start")
    test_missing_table_schema()
    print("[test_eda] missing_table OK")
    test_numeric_seven_number()
    print("[test_eda] seven-number OK")
    print("[test_eda] all passed")


if __name__ == "__main__":
    main()