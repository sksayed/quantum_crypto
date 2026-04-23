from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _load_dataframe(data_path: Path) -> pd.DataFrame:
    """Load either CSV or parquet based on file extension."""
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(data_path)
    raise ValueError(f"Unsupported input format: {data_path}")


def _print_label_diagnostics(df: pd.DataFrame, stage: str, top_n: int = 20) -> None:
    """Print label statistics to help compare notebook vs script runs."""
    labels = df["label"].astype(str)
    total_rows = len(labels)
    nunique = labels.nunique()
    test_rows = int(labels.str.endswith("_test").sum())
    train_rows = int(labels.str.endswith("_train").sum())
    other_rows = total_rows - test_rows - train_rows

    print(f"\n[{stage}] label diagnostics")
    print(f"- rows: {total_rows:,}")
    print(f"- unique labels: {nunique}")
    print(f"- *_test rows: {test_rows:,}")
    print(f"- *_train rows: {train_rows:,}")
    print(f"- other rows: {other_rows:,}")
    print(f"- top {top_n} labels by count:")
    print(labels.value_counts().head(top_n).to_string())


def _print_split_diagnostics(
    y_train_idx: pd.Series | list | tuple,
    y_test_idx: pd.Series | list | tuple,
    classes: list[str],
    top_n: int = 20,
) -> None:
    """Print class distribution in train/test split using encoded labels."""
    train_counts = pd.Series(y_train_idx).value_counts().sort_index()
    test_counts = pd.Series(y_test_idx).value_counts().sort_index()
    train_named = {classes[i]: int(train_counts.get(i, 0)) for i in range(len(classes))}
    test_named = {classes[i]: int(test_counts.get(i, 0)) for i in range(len(classes))}

    print("\n[Split diagnostics] top classes in train set")
    print(pd.Series(train_named).sort_values(ascending=False).head(top_n).to_string())
    print("\n[Split diagnostics] top classes in test set")
    print(pd.Series(test_named).sort_values(ascending=False).head(top_n).to_string())


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and labels based on notebook logic."""
    if "label" not in df.columns:
        raise ValueError("Input data must contain a 'label' column.")

    if "Source_IP" in df.columns:
        df["Source_IP"] = df["Source_IP"].astype(str)
    if "Destination_IP" in df.columns:
        df["Destination_IP"] = df["Destination_IP"].astype(str)

    # Keep parity with notebook preprocessing where this helper column exists.
    if "Severity_Score" not in df.columns:
        severity_scores = {
            "Benign": 0,
            "Reconnaissance": 2,
            "DoS": 3,
            "Mirai": 4,
            "Spoofing": 3,
            "Web-based": 2,
            "Bruteforce": 2,
        }
        df["Severity_Score"] = df["label"].map(severity_scores).fillna(1).astype(int)

    drop_cols = ["label", "Severity_Score", "Source_IP", "Destination_IP"]
    existing_drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=existing_drop_cols)
    y = df["label"]

    # Coerce non-numeric feature columns if any remain.
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM with notebook-aligned preprocessing."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to input CSV/parquet. Defaults to data/CIC_IoMT_2024_WiFi_MQTT_all.parquet.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    data_path = args.data_path or project_root / "data" / "CIC_IoMT_2024_WiFi_MQTT_all.parquet"

    models_dir = project_root / "models/lgbm_based_on_subroto_vai"
    outputs_dir = project_root / "outputs/lgbm_based_on_subroto_vai"
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    print(f"Loading data: {data_path}")
    df = _load_dataframe(data_path)
    _print_label_diagnostics(df, stage="Loaded dataset")

    X_df, y_raw = _prepare_features(df.copy())
    all_cols = list(X_df.columns)

    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y_raw)

    # Notebook-style split:
    # 70% train, 15% validation, 15% test (stratified).
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )
    X_test_df = X_temp_df
    y_test = y_temp

    print(f"Train : {X_train_df.shape[0]} samples (70%)")
    print(f"Test  : {X_test_df.shape[0]} samples (30%)")
    _print_split_diagnostics(y_train, y_test, classes=label_enc.classes_.tolist())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    num_classes = len(label_enc.classes_)
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
    }

    # Match notebook exactly: create validation split from training portion only.
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_v, label=y_v, reference=train_data)

    print("Training LightGBM...")
    t0 = time.perf_counter()
    model = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=100,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(-1),
        ],
    )
    train_time_sec = time.perf_counter() - t0

    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    precision, recall, f1s, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    report_txt = classification_report(
        y_test, y_pred, target_names=label_enc.classes_, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    model_path = models_dir / "lightgbm_model.txt"
    label_enc_path = models_dir / "label_encoder.pkl"
    scaler_path = models_dir / "standard_scaler.pkl"
    model.save_model(str(model_path))
    joblib.dump(label_enc, label_enc_path)
    joblib.dump(scaler, scaler_path)

    report_path = outputs_dir / "lightgbm_classification_report.txt"
    metrics_path = outputs_dir / "lightgbm_metrics.json"
    cm_img_path = outputs_dir / "cm_lightgbm.png"
    per_class_csv_path = outputs_dir / "lightgbm_per_class_metrics.csv"
    fi_csv_path = outputs_dir / "lightgbm_feature_importance.csv"
    fi_img_path = outputs_dir / "feature_importance_lightgbm.png"

    report_path.write_text(report_txt, encoding="utf-8")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "train_time_sec": train_time_sec,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "classes": label_enc.classes_.tolist(),
        "params": params,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_class_df = pd.DataFrame(
        {
            "class": label_enc.classes_,
            "precision": precision,
            "recall": recall,
            "f1_score": f1s,
        }
    )
    per_class_df.to_csv(per_class_csv_path, index=False)

    fi_df = pd.DataFrame(
        {
            "feature": all_cols,
            "importance_gain": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    fi_df.to_csv(fi_csv_path, index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=label_enc.classes_,
        yticklabels=label_enc.classes_,
        cmap="Blues",
    )
    plt.title("Confusion Matrix - LightGBM")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(cm_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_fi = fi_df.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_fi["feature"], top_fi["importance_gain"])
    plt.title("Top 20 Feature Importances (Gain) - LightGBM")
    plt.xlabel("Importance (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(fi_img_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nTraining complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Label encoder saved to: {label_enc_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Reports saved in: {outputs_dir}")
    print("\nClassification Report:")
    print(report_txt)


if __name__ == "__main__":
    main()
