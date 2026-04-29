"""
Ziyi Wang
CSE 163
Tests for the project.
Sanity check on data loading/label building.
Verify that EDA utils return correctly formatted tables
(eda_utils). Sanity check on machine learning splitting
and training processes using synthetic data (modeling_dt).
Run chi-square and logit tests.
"""


from typing import Final
import os
import shutil
import tempfile

import pandas as pd

import brfss_pipeline as brf
import eda_utils as eda
import modeling_dt as mdl
import plotting as pltmod
import validity_tests as vtest


TMP_PREFIX: Final[str] = "cse163_tests_"


def _make_temp_dir() -> str:
    """
    Create a temp dir and return its path.
    """
    return tempfile.mkdtemp(prefix=TMP_PREFIX)


def _cleanup_dir(path: str) -> None:
    """
    Remove a directory tree if it exists.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)


def _assert_between(x: float, lo: float, hi: float) -> None:
    """
    Assert lo <= x <= hi.
    """
    assert lo <= x <= hi, f"value {x} not in [{lo}, {hi}]"


def _synthetic_brfss() -> pd.DataFrame:
    """
    Tiny BRFSS-like table to exercise the pipeline.
    """
    data = {
        "Diabetes_012": [0, 2, 0, 2, 1, 0, 2, 0, 0, 2],
        "BMI": [22.0, 28.5, 31.0, 35.2, 26.0, 24.1, 40.0, 18.5, 27.0, 33.3],
        "HighBP": [0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        "GenHlth": [2, 3, 4, 5, 2, 2, 4, 1, 3, 5],
        "PhysHlth": [1, 4, 6, 10, 0, 2, 12, 0, 3, 8],
        "Age": [3, 7, 9, 10, 5, 2, 11, 1, 6, 8],
        "Income": [2, 4, 8, 7, 5, 1, 6, 2, 3, 8],
        "Sex": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
        "CholCheck": [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        "AnyHealthcare": [1, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    }
    df = pd.DataFrame(data)
    df["Diabetes_binary"] = (df["Diabetes_012"] == 2).astype(int)
    df["Age_grp"] = df["Age"].map(brf.AGE_LABELS)
    df["Income_grp"] = df["Income"].map(brf.INCOME_LABELS)
    df["Sex_label"] = df["Sex"].map(brf.SEX_LABELS)
    return df


def test_load_brfss_constructs_labels() -> None:
    """
    Loader creates Diabetes_binary and mapped label columns.
    """
    tmp_dir = _make_temp_dir()
    csv_path = os.path.join(tmp_dir, "mini_brfss.csv")

    df = _synthetic_brfss().drop(
        columns=["Diabetes_binary", "Age_grp", "Income_grp", "Sex_label"]
    )
    df.to_csv(csv_path, index=False)

    loaded = brf.load_brfss(csv_path)
    assert "Diabetes_binary" in loaded.columns
    assert "Age_grp" in loaded.columns
    assert "Income_grp" in loaded.columns
    assert "Sex_label" in loaded.columns

    expected = (loaded["Diabetes_012"] == 2).astype(int)
    assert loaded["Diabetes_binary"].equals(expected)

    _cleanup_dir(tmp_dir)


def test_missing_table_and_seven_number() -> None:
    """
    EDA helpers: schema + basic value sanity.
    """
    df = _synthetic_brfss()

    miss = eda.missing_table(df)
    need = {"column", "n_missing", "pct_missing"}
    assert need.issubset(set(miss.columns)), f"missing_table cols={list(miss.columns)}"
    assert len(miss) == df.shape[1]
    assert miss["pct_missing"].between(0, 100).all()

    cols = ["BMI", "HighBP", "GenHlth", "PhysHlth", "Age", "Income"]
    summ = eda.numeric_seven_number(df, cols)

    assert set(summ.index).issubset(set(cols)), f"rows={list(summ.index)} expected⊆{cols}"

    expected = {"mean", "std", "min", "q1", "median", "q3", "max"}
    got = set(map(str, summ.columns))
    missing = expected - got
    assert not missing, (
        "numeric_seven_number missing cols: "
        f"{sorted(missing)}; got={sorted(got)}"
    )


def test_plotting_saves_files() -> None:
    """
    Plot functions should write PNGs to disk (smoke test).
    """
    df = _synthetic_brfss()
    fig_dir = _make_temp_dir()

    pltmod.plot_brfss_income_prevalence(df, fig_dir)
    pltmod.plot_brfss_age_prevalence(df, fig_dir)

    p_bmi = os.path.join(fig_dir, "prev_bmi.png")
    pltmod.plot_prev_by_bmi_bins(df, "Diabetes_binary", p_bmi)

    p_rr = os.path.join(fig_dir, "rr_binary.png")
    pltmod.plot_risk_ratio_binary(
        df, ["HighBP", "AnyHealthcare"], "Diabetes_binary", p_rr
    )

    p_imp = os.path.join(fig_dir, "imp.png")
    names = ["BMI", "HighBP", "GenHlth", "PhysHlth", "Age", "Income"]
    importances = [0.3, 0.2, 0.15, 0.12, 0.11, 0.12]
    pltmod.plot_feature_importances(importances, names, p_imp, top_k=6)

    wrote = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
    assert len(wrote) >= 4

    _cleanup_dir(fig_dir)


def test_model_split_and_train() -> None:
    """
    Split/train/predict produces 0/1 labels and correct sizes.
    """
    df = _synthetic_brfss()
    feats = ["BMI", "HighBP", "GenHlth", "PhysHlth", "Age", "Income"]

    X_tr, X_te, y_tr, y_te = mdl.split_data(df[feats], df["Diabetes_binary"])
    assert len(X_tr) + len(X_te) == len(df)
    assert len(y_tr) + len(y_te) == len(df)

    model = mdl.train_decision_tree(X_tr, y_tr)
    preds = model.predict(X_te)
    assert len(preds) == len(y_te)
    for p in preds:
        assert p in (0, 1)


def test_validity_chi2() -> None:
    """
    Chi-square on induced association should usually be significant.
    """
    df = _synthetic_brfss().copy()
    df.loc[df["Income"] <= 2, "Diabetes_binary"] = 1
    df.loc[df["Income"] >= 7, "Diabetes_binary"] = 0

    res = vtest.chi2_test(df, "Income_grp", "Diabetes_binary")
    assert {"chi2", "dof", "p", "table"}.issubset(res.keys())
    _assert_between(float(res["p"]), 0.0, 1.0)
    assert float(res["p"]) <= 0.2


def test_validity_logit() -> None:
    """
    Logit summary exists; p-values lie within [0, 1].
    """
    df = _synthetic_brfss().copy()
    df["Diabetes_binary"] = (
        (df["Age"] >= 8).astype(int) | (df["Income"] <= 2).astype(int)
    ).astype(int)

    summ = vtest.logit_income_age_sex(df)
    assert isinstance(summ, pd.DataFrame)
    assert {"param", "coef", "pval"}.issubset(summ.columns)
    assert len(summ) >= 3
    for p in summ["pval"].astype(float).tolist():
        _assert_between(p, 0.0, 1.0)


def main() -> None:
    """
    Run all tests; any failed assert will stop execution.
    """
    print("[tests] start")
    test_load_brfss_constructs_labels()
    print("[tests] load_brfss OK")

    test_missing_table_and_seven_number()
    print("[tests] EDA helpers OK")

    test_plotting_saves_files()
    print("[tests] plotting OK")

    test_model_split_and_train()
    print("[tests] modeling OK")

    test_validity_chi2()
    test_validity_logit()
    print("[tests] validity OK")

    print("[tests] all passed")


if __name__ == "__main__":
    main()