"""
models/train_evaluate.py
Train and evaluate classifiers on IBC features.
Supports KNN, SVM, RF, LightGBM with subject-wise splitting.
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix  # FAR/FRR용

# --------------------------------------------------------------------------
N_SPLITS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 999]

# --------------------------------------------------------------------------
def load_data(features_csv: Path, labels_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    X = pd.read_csv(features_csv).values
    y = pd.read_csv(labels_csv)["subject_id"].values  # assuming column 'subject_id'
    return X, y


def train_model(X_train, y_train, model_name: str):
    if model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=1)
    elif model_name == "svm":
        model = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
    elif model_name == "rf":
        model = RandomForestClassifier(n_estimators=500, random_state=42)
    elif model_name == "lgbm":
        model = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # FAR, FRR (multi-class average)
    cm = confusion_matrix(y_test, y_pred)
    far = []
    frr = []
    for i in range(len(cm)):
        fp = np.sum(cm[:, i]) - cm[i, i]
        fn = np.sum(cm[i, :]) - cm[i, i]
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        tp = cm[i, i]
        far.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        frr.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
    far_avg = np.mean(far)
    frr_avg = np.mean(frr)

    # EER (multi-class: average over one-vs-rest)
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    eer_list = []
    for i in range(y_test_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        eer_list.append(eer)
    eer_avg = np.mean(eer_list)

    return {"accuracy": acc, "far": far_avg, "frr": frr_avg, "eer": eer_avg}


def train_and_evaluate(X: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    results = {"accuracy": [], "far": [], "frr": [], "eer": []}
    last_model = None
    for seed in RANDOM_SEEDS:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups=y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = train_model(X_train, y_train, model_name)
        last_model = model  # update with last model

        eval_results = evaluate_model(model, X_test, y_test)

        for k, v in eval_results.items():
            results[k].append(v)

    # Average metrics
    avg_results = {k: np.mean(v) for k, v in results.items()}
    return avg_results, last_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate IBC classifiers.")
    parser.add_argument("--features", type=Path, required=True, help="Feature CSV path")
    parser.add_argument("--labels", type=Path, required=True, help="Label CSV path")
    parser.add_argument("--model", required=True, choices=["knn", "svm", "rf", "lgbm"])
    parser.add_argument("--output_model", type=Path, required=True, help="Save model pickle")
    parser.add_argument("--output_results", type=Path, required=True, help="Save JSON results")
    args = parser.parse_args()

    X, y = load_data(args.features, args.labels)
    results, model = train_and_evaluate(X, y, args.model)

    # Save results
    with open(args.output_results, "w") as f:
        json.dump(results, f, indent=4)

    # Save model
    with open(args.output_model, "wb") as f:
        pickle.dump(model, f)

    print(f"Results: {results}")
    print(f"Model saved to {args.output_model}")
    print(f"Results saved to {args.output_results}")


if __name__ == "__main__":
    main()
