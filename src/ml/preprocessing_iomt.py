from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ml.train_iomt_gbdt_fs import DataLoaderPreprocessor, print_status
from src.ml.train_iomt_two_layer import clean_labels


# Column lists tuned for CIC_IoMT_2024_WiFi_MQTT_*.parquet.
# Those files (train/test) are fully numeric (no string/IP/text columns),
# so we leave these lists empty by default. They are kept only so that you
# can easily plug in dataset-specific overrides later if you add mixed-type features.
CATEGORICAL_SAFE: List[str] = []

# High-cardinality / text columns to drop. For the current CIC-IoMT parquet
# files there are none, so this list is empty.
EXCLUDE_COLUMNS: List[str] = []


def _load_parquet_any(data_path: str | Path, label_col: str, sample_size: int | None) -> pd.DataFrame:
    """Reuse the existing loader to read a parquet file or directory."""
    loader = DataLoaderPreprocessor(
        label_col=label_col,
        drop_cols=[],
        encode_categoricals=False,
    )
    return loader.load_parquet(Path(data_path), sample_size=sample_size)


def _build_preprocessor(
    X: pd.DataFrame,
    categorical_safe: List[str] | None = None,
    exclude_cols: List[str] | None = None,
    max_cat_cardinality: int = 40,
) -> Tuple[pd.DataFrame, List[str], ColumnTransformer, List[str], List[str]]:
    """Construct a ColumnTransformer + feature name list.

    Returns:
        X_clean: dataframe after dropping excluded columns
        feature_names: list of output feature names after transformation
        preprocessor: fitted ColumnTransformer
        num_cols: numeric columns used
        cat_cols: categorical columns used
    """
    exclude_cols = exclude_cols or []
    categorical_safe = categorical_safe or []

    # Drop explicit exclude columns if they exist
    drop_cols = [c for c in exclude_cols if c in X.columns]
    if drop_cols:
        print_status(f"Dropping {len(drop_cols)} high-cardinality / text columns: {drop_cols}", level=1)
        X = X.drop(columns=drop_cols)

    # Identify numeric and candidate categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = [c for c in categorical_safe if c in X.columns]

    # Add low-cardinality leftover object columns as categoricals
    for c in X.select_dtypes(exclude=[np.number]).columns:
        if c in cat_cols or c in exclude_cols:
            continue
        if X[c].nunique(dropna=False) <= max_cat_cardinality:
            cat_cols.append(c)

    print_status(
        f"Preprocessing: {len(num_cols)} numeric cols, {len(cat_cols)} categorical cols",
        level=1,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=0.01,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return X, num_cols, cat_cols, preprocessor


def load_and_prepare_binary(
    data_path: str | Path,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: int | None = None,
    base_outdir: str = "models",
    model_name: str = "lgbm_binary",
    test_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Pipeline, Dict[int, str]]:
    """Binary preprocessing: attack vs normal, with sklearn-style pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, preprocessor, class_map
    """
    train_df = _load_parquet_any(data_path, label_col=label_col, sample_size=sample_size)

    if label_col not in train_df.columns:
        raise ValueError(f"Target column '{label_col}' not found in dataset")

    if test_path is None:
        # Single dataset mode: split into train/test internally.
        df = train_df
        y_raw = df[label_col].astype(str).str.strip()
        classes, y = np.unique(y_raw, return_inverse=True)
        class_map: Dict[int, str] = {int(i): cls for i, cls in enumerate(classes)}

        X = df.drop(columns=[label_col])

        X, num_cols, cat_cols, preprocessor = _build_preprocessor(
            X,
            categorical_safe=CATEGORICAL_SAFE,
            exclude_cols=EXCLUDE_COLUMNS,
        )

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        print_status(
            f"[preprocessing-binary] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols...",
            level=1,
        )
        X_train = preprocessor.fit_transform(X_train_df)

        # Measure preprocessing latency on the test set (per row), approximating
        # online inference cost.
        t0 = time.time()
        X_test = preprocessor.transform(X_test_df)
        elapsed = time.time() - t0
        per_row = elapsed / max(1, len(X_test_df))
        print_status(
            f"[preprocessing-binary] transform latency: {per_row:.6f} s per row "
            f"({elapsed:.3f}s for {len(X_test_df):,} rows)",
            level=1,
        )
    else:
        # External test set mode: use train_df for fitting, test_df for evaluation.
        test_df = _load_parquet_any(test_path, label_col=label_col, sample_size=None)
        if label_col not in test_df.columns:
            raise ValueError(f"Target column '{label_col}' not found in test dataset")

        y_train_raw = train_df[label_col].astype(str).str.strip()
        y_test_raw = test_df[label_col].astype(str).str.strip()

        classes, y_train = np.unique(y_train_raw, return_inverse=True)
        class_map = {int(i): cls for i, cls in enumerate(classes)}
        class_to_index = {cls: i for i, cls in enumerate(classes)}

        # Map test labels using training classes; unseen labels become -1
        y_test = np.array([class_to_index.get(lbl, -1) for lbl in y_test_raw])

        X_train_df = train_df.drop(columns=[label_col])
        X_test_df = test_df.drop(columns=[label_col])

        X_train_df, num_cols, cat_cols, preprocessor = _build_preprocessor(
            X_train_df,
            categorical_safe=CATEGORICAL_SAFE,
            exclude_cols=EXCLUDE_COLUMNS,
        )

        print_status(
            f"[preprocessing-binary] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols "
            f"with external test set of {len(X_test_df):,} rows...",
            level=1,
        )
        X_train = preprocessor.fit_transform(X_train_df)

        t0 = time.time()
        X_test = preprocessor.transform(X_test_df)
        elapsed = time.time() - t0
        original_n = len(y_test)
        # Drop test samples whose labels were unseen during training (mapped to -1)
        keep_mask = y_test != -1
        if not np.all(keep_mask):
            removed = int((~keep_mask).sum())
            print_status(
                f"[preprocessing-binary] filtered {removed} test rows with unseen labels (-1)",
                level=1,
            )
            X_test = X_test[keep_mask]
            y_test = y_test[keep_mask]
        per_row = elapsed / max(1, original_n)
        print_status(
            f"[preprocessing-binary] transform latency (external test): {per_row:.6f} s per row "
            f"({elapsed:.3f}s for {original_n:,} rows)",
            level=1,
        )
    print_status(f"[preprocessing-binary] transformed test: {X_test.shape[0]} rows", level=1)

    feature_names: List[str] = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_out)

    # Save label map alongside model artifacts so it can be reused at inference time
    model_dir = os.path.join(base_outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    return X_train, X_test, y_train, y_test, feature_names, preprocessor, class_map


def load_and_prepare_multiclass(
    data_path: str | Path,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    sample_size: int | None = None,
    base_outdir: str = "models_mc",
    model_name: str = "lgbm_mc",
    test_path: str | Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Pipeline, Dict[int, str]]:
    """Multiclass preprocessing: factorize label column to 0..K-1 with sklearn pipeline.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, preprocessor, class_map
    """
    train_df = _load_parquet_any(data_path, label_col=label_col, sample_size=sample_size)

    if label_col not in train_df.columns:
        raise ValueError(f"Target column '{label_col}' not found in dataset")

    if test_path is None:
        # Single dataset mode: internal train/test split.
        df = train_df
        # For CIC-IoMT labels like ARP_Spoofing_train/_test collapse suffixes
        # and Benign* consistently so train/test label spaces align.
        y_raw = clean_labels(df[label_col])
        classes, y = np.unique(y_raw, return_inverse=True)
        class_map: Dict[int, str] = {int(i): cls for i, cls in enumerate(classes)}

        X = df.drop(columns=[label_col])

        X, num_cols, cat_cols, preprocessor = _build_preprocessor(
            X,
            categorical_safe=CATEGORICAL_SAFE,
            exclude_cols=EXCLUDE_COLUMNS,
        )

        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        print_status(
            f"[preprocessing-mc] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols...",
            level=1,
        )
        X_train = preprocessor.fit_transform(X_train_df)

        t0 = time.time()
        X_test = preprocessor.transform(X_test_df)
        elapsed = time.time() - t0
        per_row = elapsed / max(1, len(X_test_df))
        print_status(
            f"[preprocessing-mc] transform latency: {per_row:.6f} s per row "
            f"({elapsed:.3f}s for {len(X_test_df):,} rows)",
            level=1,
        )
        print_status(f"[preprocessing-mc] transformed test: {X_test_df.shape[0]} rows", level=1)
    else:
        # External test set mode.
        test_df = _load_parquet_any(test_path, label_col=label_col, sample_size=None)
        if label_col not in test_df.columns:
            raise ValueError(f"Target column '{label_col}' not found in test dataset")

        # Clean labels in both train and test (strip _train/_test, collapse Benign*)
        y_train_raw = clean_labels(train_df[label_col])
        y_test_raw = clean_labels(test_df[label_col])

        classes, y_train = np.unique(y_train_raw, return_inverse=True)
        class_map = {int(i): cls for i, cls in enumerate(classes)}
        class_to_index = {cls: i for i, cls in enumerate(classes)}

        # Map test labels using training label set; unseen labels become -1
        y_test = np.array([class_to_index.get(lbl, -1) for lbl in y_test_raw])

        X_train_df = train_df.drop(columns=[label_col])
        X_test_df = test_df.drop(columns=[label_col])

        X_train_df, num_cols, cat_cols, preprocessor = _build_preprocessor(
            X_train_df,
            categorical_safe=CATEGORICAL_SAFE,
            exclude_cols=EXCLUDE_COLUMNS,
        )

        print_status(
            f"[preprocessing-mc] fit on {X_train_df.shape[0]} rows, {X_train_df.shape[1]} cols "
            f"with external test set of {len(X_test_df):,} rows...",
            level=1,
        )
        X_train = preprocessor.fit_transform(X_train_df)

        t0 = time.time()
        X_test = preprocessor.transform(X_test_df)
        elapsed = time.time() - t0
        original_n = len(y_test)
        keep_mask = y_test != -1
        if not np.all(keep_mask):
            removed = int((~keep_mask).sum())
            print_status(
                f"[preprocessing-mc] filtered {removed} test rows with unseen labels (-1)",
                level=1,
            )
            X_test = X_test[keep_mask]
            y_test = y_test[keep_mask]
        per_row = elapsed / max(1, original_n)
        print_status(
            f"[preprocessing-mc] transform latency (external test): {per_row:.6f} s per row "
            f"({elapsed:.3f}s for {original_n:,} rows)",
            level=1,
        )
        print_status(f"[preprocessing-mc] transformed test: {X_test.shape[0]} rows", level=1)

    feature_names: List[str] = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(cat_out)

    model_dir = os.path.join(base_outdir, "models", model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    return X_train, X_test, y_train, y_test, feature_names, preprocessor, class_map

