"""
Two-layer intrusion detection: LGBM (Benign vs Attack) + CatBoost (attack type).

Layer 1: LightGBM binary classifier — Benign vs Attack.
Layer 2: CatBoost multiclass classifier — predicts attack type (only for samples
         predicted as Attack by layer 1).

Usage:
  python -m src.ml.train_iomt_two_layer \
    --data-path data \
    --label-col label \
    --output-dir models/iomt_two_layer
"""

from __future__ import annotations

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

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring disabled.")

# Reuse data loading from the main trainer (same prepare signature: 4 return values)
from src.ml.train_iomt_gbdt import (
    DataLoaderPreprocessor,
    get_memory_usage,
    print_status,
)

# Benign class names are those containing this substring (case-insensitive)
BENIGN_KEYWORD = "Benign"

# Suffixes to strip from labels to avoid train/test leakage (e.g. ARP_Spoofing_train -> ARP_Spoofing)
LABEL_SUFFIXES = ("_train", "_test")


def clean_labels(series: pd.Series) -> pd.Series:
    """Strip _train/_test suffixes and collapse Benign* to 'Benign'. Apply globally after load."""
    def _clean(s: str) -> str:
        s = str(s).strip()
        if BENIGN_KEYWORD in s:
            return "Benign"
        for suffix in LABEL_SUFFIXES:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
                break
        return s

    return series.astype(str).apply(_clean)


def is_benign_label(series: pd.Series) -> np.ndarray:
    """True for benign, False for attack."""
    return series.astype(str).str.contains(BENIGN_KEYWORD, case=False, na=False).values


def get_attack_classes(all_labels: list[str]) -> list[str]:
    """Return sorted list of attack class names (non-benign)."""
    return sorted([c for c in all_labels if BENIGN_KEYWORD not in c])


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to smallest dtype that fits to reduce RAM."""
    print_status(f"Original RAM usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", level=1)
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
    print_status(f"Reduced RAM usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", level=1)
    return df


def train_lgbm_binary(
    X_train: pd.DataFrame,
    y_binary_train: np.ndarray,
    X_val: pd.DataFrame,
    y_binary_val: np.ndarray,
    categorical_cols: list[str],
    n_estimators: int = 500,
    max_depth: int = 8,
    num_leaves: int = 63,
    learning_rate: float = 0.02,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_data_in_leaf: int = 15,
    early_stopping_rounds: int = 50,
):
    """Train binary LightGBM: 0 = Benign, 1 = Attack."""
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    print_status("Training Layer 1: LightGBM binary (Benign vs Attack)...")
    start = time.time()
    callbacks = [log_evaluation(period=10)]
    if early_stopping_rounds > 0:
        callbacks.append(early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=min_data_in_leaf,
        class_weight="balanced",
        n_jobs=-1,
        verbose=1,
        force_row_wise=True,
    )
    model.fit(
        X_train,
        y_binary_train,
        eval_set=[(X_val, y_binary_val)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
        categorical_feature=categorical_cols if categorical_cols else "auto",
    )
    print_status(f"Layer 1 training completed in {time.time() - start:.2f}s", level=1)
    return model


def train_catboost_attack(
    X_attack: pd.DataFrame,
    y_attack: np.ndarray,
    X_val_attack: pd.DataFrame,
    y_val_attack: np.ndarray,
    num_classes: int,
    categorical_cols: list[str] | None = None,
    n_estimators: int = 500,
    max_depth: int = 8,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 50,
    thread_count: int = 0,
    max_ctr_complexity: int = 1,
):
    """Train CatBoost multiclass for attack types only."""
    from catboost import CatBoostClassifier, Pool

    print_status("Training Layer 2: CatBoost multiclass (attack types)...")
    start = time.time()
    # Auto-configure thread count as (max_threads - 4) when not explicitly set
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
    cat_features = [c for c in (categorical_cols or []) if c in X_attack.columns]
    if cat_features:
        print_status(f"CatBoost categorical features: {len(cat_features)}", level=1)
    print_status(f"CatBoost thread_count: {thread_count}, max_ctr_complexity: {max_ctr_complexity}", level=1)

    model = CatBoostClassifier(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=learning_rate,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        early_stopping_rounds=early_stopping_rounds if early_stopping_rounds else None,
        verbose=10,
        random_seed=42,
        thread_count=thread_count,
        max_ctr_complexity=max_ctr_complexity,
    )
    train_pool = Pool(X_attack, y_attack, cat_features=cat_features if cat_features else None)
    val_pool = Pool(X_val_attack, y_val_attack, cat_features=cat_features if cat_features else None)

    print_status("Pools created. Deleting Pandas copies to free RAM for training...", level=1)
    del X_attack, y_attack, X_val_attack, y_val_attack
    gc.collect()

    model.fit(train_pool, eval_set=val_pool)
    print_status(f"Layer 2 training completed in {time.time() - start:.2f}s", level=1)
    return model


def predict_two_layer(
    lgbm_model,
    catboost_model,
    X: pd.DataFrame,
    attack_encoder: LabelEncoder,
    attack_threshold: float = 0.5,
) -> np.ndarray:
    """Predict: Benign if Layer 1 attack prob < threshold, else CatBoost attack class."""
    proba = lgbm_model.predict_proba(X)
    if proba.ndim == 1 or proba.shape[1] == 1:
        attack_proba = proba.ravel()
    else:
        attack_proba = proba[:, 1]
    out = np.empty(len(X), dtype=object)
    attack_mask = attack_proba >= attack_threshold
    out[~attack_mask] = "Benign"
    if attack_mask.any():
        X_attack = X.iloc[np.where(attack_mask)[0]]
        attack_pred = catboost_model.predict(X_attack)
        if hasattr(attack_pred, "ravel"):
            attack_pred = attack_pred.ravel()
        out[attack_mask] = attack_encoder.inverse_transform(attack_pred.astype(int))
    return out


def evaluate_two_layer(
    lgbm_model,
    catboost_model,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    attack_encoder: LabelEncoder,
    attack_threshold: float,
) -> str:
    """Predict and build classification report. Labels are already cleaned globally (Fix 1)."""
    y_pred = predict_two_layer(
        lgbm_model,
        catboost_model,
        X_test,
        attack_encoder,
        attack_threshold=attack_threshold,
    )
    y_test_arr = np.asarray(y_test_raw).astype(str)
    labels = ["Benign"] + list(attack_encoder.classes_)
    report = classification_report(
        y_test_arr,
        y_pred,
        labels=labels,
        zero_division=0,
        digits=4,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-layer IDS: LGBM (Benign vs Attack) + CatBoost (attack type)"
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
    parser.add_argument("--sample-size", type=int, default=None, help="Subsample for quick runs")
    # Layer 1 (LGBM binary)
    parser.add_argument("--lgbm-n-estimators", type=int, default=500, dest="lgbm_n_estimators")
    parser.add_argument("--lgbm-max-depth", type=int, default=8, dest="lgbm_max_depth")
    parser.add_argument("--lgbm-num-leaves", type=int, default=31, dest="lgbm_num_leaves")
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05, dest="lgbm_learning_rate")
    parser.add_argument(
        "--lgbm-early-stopping",
        type=int,
        default=50,
        dest="lgbm_early_stopping",
        help="Layer 1 early stopping rounds (default: 50)",
    )
    # Layer 2 (CatBoost)
    parser.add_argument("--catboost-iterations", type=int, default=500, dest="catboost_iterations")
    parser.add_argument("--catboost-depth", type=int, default=8, dest="catboost_depth")
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05, dest="catboost_lr")
    parser.add_argument(
        "--catboost-early-stopping",
        type=int,
        default=50,
        dest="catboost_early_stopping",
        help="Layer 2 early stopping rounds (default: 50)",
    )
    parser.add_argument(
        "--layer1-attack-threshold",
        type=float,
        default=0.5,
        help="Threshold on Layer 1 attack probability to route samples to Layer 2 (default: 0.5)",
    )
    parser.add_argument(
        "--catboost-thread-count",
        type=int,
        default=0,
        dest="catboost_thread_count",
        help="CatBoost thread count (0 = auto: max_threads - 4)",
    )
    parser.add_argument(
        "--catboost-max-ctr-complexity",
        type=int,
        default=1,
        dest="catboost_max_ctr_complexity",
        help="CatBoost max CTR complexity; 1 avoids categorical combination explosion (default: 1)",
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
        import lightgbm  # noqa: F401
    except ImportError as e:
        raise SystemExit("LightGBM is required: pip install lightgbm") from e
    try:
        import catboost  # noqa: F401
    except ImportError as e:
        raise SystemExit("CatBoost is required: pip install catboost") from e

    print_status("=" * 60)
    print_status("CIC-IoMT-2024 Two-Layer Intrusion Detection")
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
    X_full, y_raw, feature_cols, categorical_cols = data_loader.prepare(
        train_df, build_maps=True
    )
    del train_df
    gc.collect()
    print_status("Reducing training data memory...", level=1)
    X_full = reduce_mem_usage(X_full)

    # Critical Fix 1: Strip _train/_test suffixes and collapse Benign* -> "Benign" globally
    print_status("Cleaning labels (strip _train/_test, collapse Benign)...", level=1)
    y_raw = clean_labels(y_raw)

    if test_df is not None:
        print_status("Preprocessing test data...")
        X_test_df, y_test_raw, _, _ = data_loader.prepare(test_df, build_maps=False)
        y_test_raw = clean_labels(y_test_raw)
        X_test_df = X_test_df.reindex(columns=feature_cols, fill_value=0)
        del test_df
        gc.collect()
        print_status("Reducing test data memory...", level=1)
        X_test_df = reduce_mem_usage(X_test_df)
    else:
        X_test_df = None
        y_test_raw = None

    # Binary labels: 0 = Benign, 1 = Attack (computed globally before splitting)
    y_binary = np.where(is_benign_label(y_raw), 0, 1)
    print_status(
        f"Benign vs Attack: {int((y_binary == 0).sum()):,} Benign, {int((y_binary == 1).sum()):,} Attack",
        level=1,
    )

    if X_test_df is None:
        print_status("Splitting train/val/test (stratify on multiclass labels)...")
        X_train, X_temp, y_raw_train, y_raw_temp = train_test_split(
            X_full,
            y_raw,
            test_size=args.test_size + args.val_size,
            random_state=args.random_seed,
            stratify=y_raw,
        )
        val_ratio = args.val_size / (args.test_size + args.val_size)
        X_val, X_test, y_raw_val, y_raw_test = train_test_split(
            X_temp,
            y_raw_temp,
            test_size=1 - val_ratio,
            random_state=args.random_seed,
            stratify=y_raw_temp,
        )
        y_bin_train = np.where(is_benign_label(y_raw_train), 0, 1)
        y_bin_val = np.where(is_benign_label(y_raw_val), 0, 1)
        y_test_raw = y_raw_test.reset_index(drop=True)
        X_test_df = X_test.reset_index(drop=True)
    else:
        print_status("Splitting train/val (stratify on multiclass labels)...")
        X_train, X_val, y_raw_train, y_raw_val = train_test_split(
            X_full,
            y_raw,
            test_size=args.val_size,
            random_state=args.random_seed,
            stratify=y_raw,
        )
        y_bin_train = np.where(is_benign_label(y_raw_train), 0, 1)
        y_bin_val = np.where(is_benign_label(y_raw_val), 0, 1)
        y_test_raw = y_test_raw.reset_index(drop=True)

    # Build attack label encoder using only train/val labels to avoid test leakage
    all_attack_labels = pd.concat([y_raw_train, y_raw_val], axis=0).astype(str)
    attack_classes = get_attack_classes(sorted(all_attack_labels.unique().tolist()))
    print_status(f"Attack types: {len(attack_classes)}", level=1)

    attack_encoder = LabelEncoder()
    attack_encoder.fit(attack_classes)

    # Attack-only subsets for CatBoost
    train_attack_mask = y_bin_train == 1
    val_attack_mask = y_bin_val == 1
    X_train_attack = X_train.loc[train_attack_mask].reset_index(drop=True)
    y_train_attack = attack_encoder.transform(
        y_raw_train.loc[train_attack_mask].astype(str)
    )
    X_val_attack = X_val.loc[val_attack_mask].reset_index(drop=True)
    y_val_attack = attack_encoder.transform(
        y_raw_val.loc[val_attack_mask].astype(str)
    )

    print_status(f"Layer 2 train size: {len(X_train_attack):,} attack samples", level=1)
    print_status(f"Layer 2 val size: {len(X_val_attack):,} attack samples", level=1)

    # Ensure categorical columns are proper pandas 'category' dtype for LightGBM
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")
            if X_test_df is not None and col in X_test_df.columns:
                X_test_df[col] = X_test_df[col].astype("category")

    lgbm_model = train_lgbm_binary(
        X_train,
        y_bin_train,
        X_val,
        y_bin_val,
        categorical_cols,
        n_estimators=args.lgbm_n_estimators,
        max_depth=args.lgbm_max_depth,
        num_leaves=args.lgbm_num_leaves,
        learning_rate=args.lgbm_learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=20,
        early_stopping_rounds=args.lgbm_early_stopping,
    )

    # Free Layer 1 and all other large data so CatBoost has maximum RAM
    del X_train, X_val, y_bin_train, y_bin_val
    del X_full, y_raw, train_attack_mask, val_attack_mask
    gc.collect()
    print_status("Freed LightGBM training data and full datasets; running gc.collect() before CatBoost.", level=1)

    catboost_model = train_catboost_attack(
        X_train_attack,
        y_train_attack,
        X_val_attack,
        y_val_attack,
        num_classes=len(attack_classes),
        categorical_cols=categorical_cols,
        n_estimators=args.catboost_iterations,
        max_depth=args.catboost_depth,
        learning_rate=args.catboost_lr,
        early_stopping_rounds=args.catboost_early_stopping,
        thread_count=args.catboost_thread_count,
        max_ctr_complexity=args.catboost_max_ctr_complexity,
    )

    print_status("Evaluating two-layer pipeline on test set...")
    report = evaluate_two_layer(
        lgbm_model,
        catboost_model,
        X_test_df,
        y_test_raw,
        attack_encoder,
        attack_threshold=args.layer1_attack_threshold,
    )
    print_status("=" * 60)
    print("\n" + report)
    print_status("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_status("Saving models and metadata...")
    lgbm_model.booster_.save_model(str(output_dir / "lgbm_binary.txt"))
    catboost_model.save_model(str(output_dir / "catboost_attack.cbm"))

    metadata = {
        "label_col": args.label_col,
        "feature_cols": feature_cols,
        "benign_keyword": BENIGN_KEYWORD,
        "attack_classes": attack_classes,
        "drop_cols": args.drop_cols,
        "encode_categoricals": args.encode_categoricals,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "layer1_attack_threshold": args.layer1_attack_threshold,
        "layer1_hyperparams": {
            "n_estimators": args.lgbm_n_estimators,
            "max_depth": args.lgbm_max_depth,
            "num_leaves": args.lgbm_num_leaves,
            "learning_rate": args.lgbm_learning_rate,
            "early_stopping_rounds": args.lgbm_early_stopping,
        },
        "layer2_hyperparams": {
            "iterations": args.catboost_iterations,
            "depth": args.catboost_depth,
            "learning_rate": args.catboost_lr,
            "early_stopping_rounds": args.catboost_early_stopping,
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # Save attack label encoder classes for inference
    with open(output_dir / "attack_classes.json", "w", encoding="utf-8") as f:
        json.dump(attack_encoder.classes_.tolist(), f, indent=2)

    print_status("Done.")
    print_status(f"Models: {output_dir / 'lgbm_binary.txt'}, {output_dir / 'catboost_attack.cbm'}")
    print_status(f"Metadata: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
