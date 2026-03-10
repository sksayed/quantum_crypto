from __future__ import annotations

"""
Standalone CatBoost multiclass intrusion detection on CIC-IoMT-2024.

This script mirrors the style of the existing IoMT trainers but uses
only a single CatBoost model for multiclass detection (no two-layer pipeline).

Example (single parquet, internal split):
  python -m src.ml.cat_boost_only \
    --data-path data/CIC_IoMT_2024_WiFi_MQTT_all.parquet \
    --label-col label \
    --output-dir outputs/iomt_catboost_only

Example (explicit train/test parquet):
  python -m src.ml.cat_boost_only \
    --train-path data/CIC_IoMT_2024_WiFi_MQTT_train.parquet \
    --test-path  data/CIC_IoMT_2024_WiFi_MQTT_test.parquet \
    --label-col label \
    --output-dir outputs/iomt_catboost_only
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.ml.train_iomt_gbdt_fs import (
    DataLoaderPreprocessor,
    get_memory_usage,
    print_status,
)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce RAM usage."""
    print_status(
        f"Original RAM usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        level=1,
    )
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != "category":
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    print_status(
        f"Reduced RAM usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        level=1,
    )
    return df


def train_catboost_multiclass(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_cols: list[str] | None,
    num_classes: int,
    iterations: int = 500,
    depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    thread_count: int = 0,
    max_ctr_complexity: int = 1,
):
    """Train a single CatBoost multiclass classifier on all classes."""
    from catboost import CatBoostClassifier, Pool

    print_status("Training CatBoost multiclass model...")

    if thread_count <= 0:
        try:
            if PSUTIL_AVAILABLE:
                total_threads = psutil.cpu_count(logical=True) or psutil.cpu_count()
            else:
                import os

                total_threads = os.cpu_count()
        except Exception:
            total_threads = None
        if not total_threads:
            thread_count = 2
        else:
            thread_count = max(1, total_threads - 4)

    cat_features = [c for c in (categorical_cols or []) if c in X_train.columns]
    if cat_features:
        print_status(f"CatBoost categorical features: {len(cat_features)}", level=1)
    print_status(f"CatBoost thread_count: {thread_count}", level=1)
    print_status(f"CatBoost max_ctr_complexity: {max_ctr_complexity}", level=1)

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds else None,
        verbose=50,
        random_seed=42,
        thread_count=thread_count,
        max_ctr_complexity=max_ctr_complexity,
        classes_count=num_classes,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_features or None)
    val_pool = Pool(X_val, y_val, cat_features=cat_features or None)

    start = time.time()
    model.fit(train_pool, eval_set=val_pool)
    print_status(
        f"CatBoost training completed in {time.time() - start:.2f}s",
        level=1,
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-model CatBoost multiclass IDS on CIC-IoMT-2024"
    )
    parser.add_argument("--data-path", help="Parquet file or directory")
    parser.add_argument("--train-path", help="Training parquet file or directory")
    parser.add_argument("--test-path", help="Testing parquet file or directory")
    parser.add_argument("--label-col", required=True, help="Label column name")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts")
    parser.add_argument("--drop-cols", nargs="*", default=[], help="Columns to drop")
    parser.add_argument("--encode-categoricals", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None)

    parser.add_argument("--catboost-iterations", type=int, default=500)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--catboost-early-stopping",
        type=int,
        default=50,
        help="Early stopping rounds (default: 50)",
    )
    parser.add_argument(
        "--catboost-thread-count",
        type=int,
        default=0,
        help="CatBoost thread count (0 = auto: max_threads - 4)",
    )
    parser.add_argument(
        "--catboost-max-ctr-complexity",
        type=int,
        default=1,
        help="Max CTR complexity (1 keeps categorical combinations simple)",
    )

    args = parser.parse_args()

    if not args.data_path and not args.train_path:
        raise SystemExit("Provide --data-path or --train-path")
    if args.data_path and args.train_path:
        raise SystemExit("Use either --data-path or --train-path, not both")
    if args.test_path and not args.train_path:
        raise SystemExit("--test-path requires --train-path")
    if args.test_size + args.val_size >= 1.0:
        raise SystemExit("test_size + val_size must be < 1.0")

    try:
        import catboost  # noqa: F401
    except ImportError as e:
        raise SystemExit("CatBoost is required: pip install catboost") from e

    print_status("=" * 60)
    print_status("CIC-IoMT-2024 Single-Model CatBoost Multiclass IDS")
    print_status("=" * 60)

    data_loader = DataLoaderPreprocessor(
        label_col=args.label_col,
        drop_cols=args.drop_cols,
        encode_categoricals=args.encode_categoricals,
    )

    if args.train_path:
        train_df = data_loader.load_parquet(Path(args.train_path), sample_size=args.sample_size)
        test_df = (
            data_loader.load_parquet(Path(args.test_path), sample_size=None)
            if args.test_path
            else None
        )
    else:
        train_df = data_loader.load_parquet(Path(args.data_path), sample_size=args.sample_size)
        test_df = None

    print_status("Preprocessing training data...")
    X_full, y_raw, feature_cols, categorical_cols, _, _ = data_loader.prepare(
        train_df,
        build_maps=True,
        null_zero_threshold=None,
    )
    del train_df
    gc.collect()
    print_status("Reducing training data memory...", level=1)
    X_full = reduce_mem_usage(X_full)

    if test_df is not None:
        print_status("Preprocessing test data (external test set)...")
        X_test_df, y_test_raw, _, _, _, _ = data_loader.prepare(
            test_df,
            build_maps=False,
            null_zero_threshold=None,
        )
        X_test_df = X_test_df.reindex(columns=feature_cols, fill_value=0)
        del test_df
        gc.collect()
        print_status("Reducing test data memory (external test)...", level=1)
        X_test_df = reduce_mem_usage(X_test_df)
    else:
        X_test_df = None
        y_test_raw = None

    print_status("Encoding labels...", level=1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw.astype(str))
    num_classes = len(label_encoder.classes_)
    print_status(f"Number of classes: {num_classes}", level=1)

    if X_test_df is None:
        print_status("Splitting train/val/test (internal split, stratify on labels)...")
        X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
            X_full,
            y_encoded,
            test_size=args.test_size + args.val_size,
            random_state=args.random_seed,
            stratify=y_encoded,
        )
        val_ratio = args.val_size / (args.test_size + args.val_size)
        X_val, X_test, y_val_raw, y_test = train_test_split(
            X_temp,
            y_temp_raw,
            test_size=1 - val_ratio,
            random_state=args.random_seed,
            stratify=y_temp_raw,
        )
        y_train = y_train_raw
        y_val = y_val_raw
        X_test_df = X_test
    else:
        print_status("Splitting train/val (external test set provided)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_encoded,
            test_size=args.val_size,
            random_state=args.random_seed,
            stratify=y_encoded,
        )
        y_test = label_encoder.transform(y_test_raw.astype(str))

    del X_full
    gc.collect()

    print_status(
        f"Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(y_test):,}",
        level=1,
    )

    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            if X_test_df is not None and col in X_test_df.columns:
                X_test_df[col] = X_test_df[col].astype("category")

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)

    cat_model = train_catboost_multiclass(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        categorical_cols=categorical_cols,
        num_classes=num_classes,
        iterations=args.catboost_iterations,
        depth=args.catboost_depth,
        learning_rate=args.catboost_learning_rate,
        early_stopping_rounds=args.catboost_early_stopping,
        thread_count=args.catboost_thread_count,
        max_ctr_complexity=args.catboost_max_ctr_complexity,
    )

    print_status("Evaluating CatBoost model on test set...")
    from catboost import Pool

    test_pool = Pool(X_test_df, label=y_test)
    y_pred_int = cat_model.predict(test_pool)
    if hasattr(y_pred_int, "ravel"):
        y_pred_int = y_pred_int.ravel()
    y_pred_int = y_pred_int.astype(int)
    y_test_int = y_test.astype(int)

    target_names = label_encoder.classes_.tolist()
    report = classification_report(
        y_test_int,
        y_pred_int,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print_status("=" * 60)
    print("\n" + report)
    print_status("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_status("Saving CatBoost model and metadata...")
    model_path = output_dir / "catboost_multiclass.cbm"
    cat_model.save_model(str(model_path))

    metadata = {
        "label_col": args.label_col,
        "feature_cols": feature_cols,
        "label_classes": target_names,
        "drop_cols": args.drop_cols,
        "encode_categoricals": args.encode_categoricals,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "catboost_hyperparams": {
            "iterations": args.catboost_iterations,
            "depth": args.catboost_depth,
            "learning_rate": args.catboost_learning_rate,
            "early_stopping_rounds": args.catboost_early_stopping,
            "thread_count": args.catboost_thread_count,
            "max_ctr_complexity": args.catboost_max_ctr_complexity,
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print_status("Done.")
    print_status(f"Model: {model_path}")
    print_status(f"Metadata: {output_dir / 'metadata.json'}")
    print_status(f"Report: {output_dir / 'classification_report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\nTraining interrupted by user", level=0)
        sys.exit(1)
    except Exception as e:
        print_status(f"\nERROR: Training failed: {type(e).__name__}: {e}", level=0)
        import traceback

        traceback.print_exc()
        sys.exit(1)

