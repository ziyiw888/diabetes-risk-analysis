"""
Ziyi Wang
CSE 163 AC
Wraps a DecisionTreeClassifier for BRFSS diabetes prediction.
"""


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """
    Train and test split with stratification.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """
    Fit a DecisionTreeClassifier with a small grid search over max_depth.
    """
    param_grid = {'max_depth': [3, 5, 7, 9, None]}
    clf = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf.best_estimator_

def evaluate_model(model: DecisionTreeClassifier,
                   X_test: pd.DataFrame, y_test: pd.Series):
    """
    Compute standard binary classification metrics on a held-out
    test set and print a concise report.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    f1  = f1_score(y_test, pred)
    cm  = confusion_matrix(y_test, pred)

    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    print("F1:", f1)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, pred, digits=3))

    return {"accuracy": acc, "roc_auc": auc, "f1": f1}

def plot_feature_importances(model: DecisionTreeClassifier,
                             feature_names: list, top_k: int = 10) -> None:
    """
    Horizontal bar plot of top-k feature importances.
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_k)

    plt.figure(figsize=(6, 4))
    plt.barh(importances['feature'], importances['importance'])
    plt.gca().invert_yaxis()
    plt.title(f"Top-{top_k} Feature Importances (Decision Tree)")
    plt.tight_layout()

def group_metrics(model: DecisionTreeClassifier,
                  X_test: pd.DataFrame,
                  y_test: pd.Series,
                  group_series: pd.Series,
                  min_n: int = 100) -> pd.DataFrame:
    """
    Computes accuracy / ROC AUC / F1 by groups.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    df = X_test.copy()
    df["y_true"] = y_test.values
    df["y_proba"] = proba
    df["y_pred"] = pred
    df["group"]  = group_series.values

    rows = []
    for g, sub in df.groupby("group"):
        if len(sub) < min_n:
            continue

        acc = accuracy_score(sub["y_true"], sub["y_pred"])

        has_pos = (sub["y_true"] == 1).any()
        has_neg = (sub["y_true"] == 0).any()
        if has_pos and has_neg:
            auc = roc_auc_score(sub["y_true"], sub["y_proba"])
        else:
            auc = float("nan")

        f1v = f1_score(sub["y_true"], sub["y_pred"])

        rows.append({
            "group": g,
            "n": len(sub),
            "accuracy": acc,
            "roc_auc": auc,
            "f1": f1v
        })

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)