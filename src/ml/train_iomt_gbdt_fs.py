"""
Train a multi-class LightGBM model with feature filtering and selection.

Steps:
1) Drop features where >= threshold are null or zero (or constant).
2) Train a model to get feature importances.
3) Keep top N features (15-20 recommended) and retrain the final model.

Usage:
  python -m src.ml.train_iomt_gbdt_fs \
    --data-path /path/to/data.parquet \
    --label-col Label \
    --output-dir models/iomt_gbdt_fs \
    --top-features 20 \
    --null-zero-threshold 0.4
"""

from __future__ import annotations

import argparse
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


def print_status(message: str, level: int = 0) -> None:
    """Print status message with indentation."""
    indent = "  " * level
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {indent}{message}", flush=True)


def get_memory_usage() -> dict:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return {"available": "N/A", "used": "N/A", "percent": "N/A"}
    try:
        mem = psutil.virtual_memory()
        return {
            "available_mb": mem.available / (1024**2),
            "used_mb": mem.used / (1024**2),
            "percent": mem.percent
        }
    except Exception:
        return {"available": "N/A", "used": "N/A", "percent": "N/A"}


class DataLoaderPreprocessor:
    """Load parquet data and produce clean feature matrices."""

    def __init__(
        self,
        label_col: str = "label",
        drop_cols: list[str] | None = None,
        encode_categoricals: bool = True,
    ) -> None:
        self.label_col = label_col
        self.drop_cols = drop_cols or []
        self.encode_categoricals = encode_categoricals
        self.categorical_cols: list[str] = []
        self.categorical_categories: dict[str, list[str]] = {}

    def load_parquet(self, data_path: Path, sample_size: int | None = None) -> pd.DataFrame:
        """Load parquet file(s) with progress tracking and optional sampling."""
        print_status(f"Loading data from: {data_path}")
        mem_before = get_memory_usage()
        print_status(
            f"Memory before loading: {mem_before.get('used_mb', 'N/A'):.1f} MB used "
            f"({mem_before.get('percent', 'N/A')}%)",
            level=1,
        )

        start_time = time.time()

        if data_path.is_dir():
            parquet_files = sorted(data_path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No .parquet files found in {data_path}")
            print_status(f"Found {len(parquet_files)} parquet file(s)", level=1)
            frames = []
            for i, p in enumerate(parquet_files, 1):
                print_status(f"Loading file {i}/{len(parquet_files)}: {p.name}", level=1)
                df_chunk = pd.read_parquet(p)
                frames.append(df_chunk)
                print_status(f"  Loaded {len(df_chunk):,} rows", level=2)
            df = pd.concat(frames, ignore_index=True)
            print_status(f"Concatenated {len(df):,} total rows", level=1)
        else:
            if not data_path.exists():
                raise FileNotFoundError(f"Data path not found: {data_path}")
            print_status(f"Reading parquet file: {data_path.name}", level=1)
            df = pd.read_parquet(data_path)
            print_status(f"Loaded {len(df):,} rows, {len(df.columns)} columns", level=1)

        # Optional sampling for memory efficiency (stratified by label if possible)
        if sample_size is not None and len(df) > sample_size:
            print_status(
                f"Sampling {sample_size:,} rows from {len(df):,} total rows",
                level=1,
            )
            try:
                label_col = self.label_col
                if label_col in df.columns:
                    _, df_sampled = train_test_split(
                        df,
                        train_size=sample_size,
                        stratify=df[label_col],
                        random_state=42,
                    )
                    df = df_sampled.reset_index(drop=True)
                    print_status(f"Stratified sampling: {len(df):,} rows", level=2)
                    label_counts = df[label_col].value_counts()
                    print_status(
                        f"Classes after sampling: {len(label_counts)} classes",
                        level=2,
                    )
                    print_status(
                        f"  Min samples per class: {label_counts.min()}, Max: {label_counts.max()}",
                        level=2,
                    )
                else:
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    print_status(f"Random sampling: {len(df):,} rows", level=2)
            except Exception as e:
                print_status(
                    f"Stratified sampling failed ({e}), using random sampling",
                    level=2,
                )
                df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            print_status(f"Sampled dataset: {len(df):,} rows", level=1)

        elapsed = time.time() - start_time
        mem_after = get_memory_usage()
        print_status(f"Data loaded in {elapsed:.2f}s", level=1)
        print_status(
            f"Memory after loading: {mem_after.get('used_mb', 'N/A'):.1f} MB used "
            f"({mem_after.get('percent', 'N/A')}%)",
            level=1,
        )
        print_status(f"Dataset size: {len(df):,} rows × {len(df.columns)} columns", level=1)

        return df

    def _build_categorical_schema(self, df: pd.DataFrame) -> None:
        self.categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
        self.categorical_categories = {}
        for col in self.categorical_cols:
            categories = pd.Series(df[col].astype(str).unique()).sort_values()
            self.categorical_categories[col] = categories.tolist()

    def _apply_categorical_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, categories in self.categorical_categories.items():
            if col not in df.columns:
                df[col] = pd.Categorical([None] * len(df), categories=categories)
            else:
                df[col] = df[col].astype(str)
                df[col] = df[col].astype("category").cat.set_categories(categories)
        return df

    def prepare(
        self,
        df: pd.DataFrame,
        build_maps: bool = False,
        null_zero_threshold: float | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict[str, dict[str, float]], list[str]]:
        print_status("Preprocessing data...")
        start_time = time.time()

        if self.label_col not in df.columns:
            raise KeyError(
                f"Label column not found: {self.label_col}. "
                f"Available columns: {list(df.columns)[:10]}..."
            )

        print_status(f"Extracting label column: {self.label_col}", level=1)
        df = df.copy()
        y = df.pop(self.label_col)
        print_status(f"Label distribution: {y.value_counts().shape[0]} unique classes", level=1)

        if self.drop_cols:
            print_status(f"Dropping {len(self.drop_cols)} columns: {self.drop_cols}", level=1)
            for col in self.drop_cols:
                if col in df.columns:
                    df = df.drop(columns=[col])

        if self.encode_categoricals:
            if build_maps:
                print_status("Building categorical schema...", level=1)
                self._build_categorical_schema(df)
                print_status(f"Found {len(self.categorical_cols)} categorical columns", level=2)
            print_status("Applying categorical schema...", level=1)
            df = self._apply_categorical_schema(df)
            non_numeric = [
                c for c in df.columns
                if df[c].dtype == "object" and c not in self.categorical_cols
            ]
            if non_numeric:
                print_status(
                    f"Dropping {len(non_numeric)} non-numeric columns: {non_numeric[:5]}...",
                    level=1,
                )
                df = df.drop(columns=non_numeric)
        else:
            non_numeric = [c for c in df.columns if df[c].dtype == "object"]
            if non_numeric:
                print_status(
                    f"Dropping {len(non_numeric)} non-numeric columns: {non_numeric[:5]}...",
                    level=1,
                )
                df = df.drop(columns=non_numeric)

        print_status("Handling infinite values and NaNs...", level=1)
        df = df.replace([np.inf, -np.inf], np.nan)

        feature_stats: dict[str, dict[str, float]] = {}
        dropped_features: list[str] = []
        if null_zero_threshold is not None:
            print_status(
                f"Filtering features with null/zero rate >= {null_zero_threshold:.0%}",
                level=1,
            )
            for col in df.columns:
                null_rate = float(df[col].isna().mean())
                zero_rate = float((df[col] == 0).mean()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                unique_count = int(df[col].nunique(dropna=True))
                feature_stats[col] = {
                    "null_rate": null_rate,
                    "zero_rate": zero_rate,
                    "unique_count": unique_count,
                }
                if max(null_rate, zero_rate) >= null_zero_threshold or unique_count <= 1:
                    dropped_features.append(col)
            if dropped_features:
                print_status(
                    f"Dropping {len(dropped_features)} low-signal features",
                    level=1,
                )
                df = df.drop(columns=dropped_features)

        # Handle all-NaN columns
        nan_cols = df.columns[df.isna().all()].tolist()
        if nan_cols:
            print_status(
                f"Warning: {len(nan_cols)} columns are all NaN, filling with 0",
                level=2,
            )
            df[nan_cols] = 0

        # Fill remaining NaNs with median
        df = df.fillna(df.median(numeric_only=True))

        feature_cols = df.columns.tolist()
        categorical_cols = (
            [c for c in self.categorical_cols if c in feature_cols]
            if self.encode_categoricals
            else []
        )
        elapsed = time.time() - start_time
        print_status(f"Preprocessing completed in {elapsed:.2f}s", level=1)
        print_status(
            f"Final feature matrix: {len(df):,} rows × {len(feature_cols)} features",
            level=1,
        )

        return df, y, feature_cols, categorical_cols, feature_stats, dropped_features


class ModelTrainer:
    """Train a LightGBM multi-class classifier."""

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        num_leaves: int,
        learning_rate: float,
        subsample: float,
        colsample_bytree: float,
        min_data_in_leaf: int = 20,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_data_in_leaf = min_data_in_leaf

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        num_classes: int,
        class_weight: str | None = None,
        early_stopping_rounds: int | None = None,
        categorical_feature: list[str] | None = None,
    ):
        from lightgbm import LGBMClassifier
        from sklearn.utils.class_weight import compute_class_weight

        print_status("Initializing LightGBM model...")
        print_status(f"Training samples: {len(X_train):,}", level=1)
        print_status(f"Validation samples: {len(X_val):,}", level=1)
        print_status(f"Number of classes: {num_classes}", level=1)

        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print_status("Class distribution in training set:", level=1)
        for cls, count in zip(unique_train[:10], counts_train[:10]):
            print_status(
                f"  Class {cls}: {count:,} samples ({count/len(y_train)*100:.2f}%)",
                level=2,
            )
        if len(unique_train) > 10:
            print_status(f"  ... and {len(unique_train)-10} more classes", level=2)

        class_weight_dict = None
        if class_weight == "balanced":
            print_status("Computing balanced class weights...", level=1)
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, weights))
            print_status(
                f"Class weights computed (min: {min(weights):.2f}, max: {max(weights):.2f})",
                level=2,
            )

        print_status("Hyperparameters:", level=1)
        print_status(f"  - n_estimators: {self.n_estimators}", level=2)
        print_status(f"  - max_depth: {self.max_depth}", level=2)
        print_status(f"  - num_leaves: {self.num_leaves}", level=2)
        print_status(f"  - learning_rate: {self.learning_rate}", level=2)
        print_status(f"  - subsample: {self.subsample}", level=2)
        print_status(f"  - colsample_bytree: {self.colsample_bytree}", level=2)
        print_status(f"  - min_data_in_leaf: {self.min_data_in_leaf}", level=2)
        if class_weight_dict:
            print_status("  - class_weight: balanced", level=2)
        if early_stopping_rounds:
            print_status(f"  - early_stopping_rounds: {early_stopping_rounds}", level=2)

        mem_before = get_memory_usage()
        print_status(
            f"Memory before training: {mem_before.get('used_mb', 'N/A'):.1f} MB",
            level=1,
        )

        model = LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_data_in_leaf,
            class_weight=class_weight_dict,
            n_jobs=-1,
            verbose=1,
            force_row_wise=True,
        )

        print_status("Starting model training...")
        start_time = time.time()

        callbacks = []
        if early_stopping_rounds and early_stopping_rounds > 0:
            from lightgbm import early_stopping
            callbacks.append(early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))
        try:
            from lightgbm import log_evaluation
            callbacks.append(log_evaluation(period=10))
        except Exception:
            print_status("LightGBM log_evaluation not available; skipping", level=1)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="multi_logloss",
            callbacks=callbacks if callbacks else None,
            categorical_feature=categorical_feature,
        )

        elapsed = time.time() - start_time
        mem_after = get_memory_usage()
        print_status(
            f"Training completed in {elapsed:.2f}s ({elapsed/60:.2f} minutes)",
            level=1,
        )
        print_status(
            f"Memory after training: {mem_after.get('used_mb', 'N/A'):.1f} MB",
            level=1,
        )
        print_status(
            f"Best iteration: {model.best_iteration_ if hasattr(model, 'best_iteration_') else 'N/A'}",
            level=1,
        )

        return model


class ModelTester:
    """Evaluate the trained model."""

    def evaluate(self, model, X_test: pd.DataFrame, y_test: np.ndarray, labels: list[str]) -> str:
        print_status(f"Evaluating model on {len(X_test):,} test samples...")
        start_time = time.time()

        y_pred = model.predict(X_test)

        elapsed = time.time() - start_time
        print_status(f"Prediction completed in {elapsed:.2f}s", level=1)

        unique_test_classes = np.unique(y_test)
        unique_pred_classes = np.unique(y_pred)
        all_present_classes = np.unique(np.concatenate([unique_test_classes, unique_pred_classes]))

        print_status(f"Classes in test set: {len(unique_test_classes)}", level=1)
        print_status(f"Classes in predictions: {len(unique_pred_classes)}", level=1)
        print_status(f"Total unique classes: {len(all_present_classes)}", level=1)

        present_label_names = [
            labels[int(cls)] for cls in all_present_classes if int(cls) < len(labels)
        ]

        print_status("Generating classification report...", level=1)
        report = classification_report(
            y_test,
            y_pred,
            labels=all_present_classes,
            target_names=present_label_names if len(present_label_names) == len(all_present_classes) else None,
            digits=4,
            zero_division=0,
        )
        return report


def save_feature_importance_plot(
    model,
    feature_cols: list[str],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Generate and save a horizontal bar chart of feature importances."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print_status(
            "matplotlib/seaborn not available; skipping feature importance plot",
            level=1,
        )
        return

    importances = model.feature_importances_
    data = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    data = data.sort_values(by="Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=data, palette="viridis")
    plt.title(f"Top {top_n} Features - CIC-IoMT-2024")
    plt.xlabel("Importance (Gain/Split)")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def select_top_features(model, feature_cols: list[str], top_features: int) -> list[str]:
    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        raise ValueError("Feature importances length does not match feature columns.")
    ranking = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    top_n = min(top_features, len(feature_cols))
    selected = ranking.head(top_n)["feature"].tolist()
    return selected


class TrainingPipeline:
    """Orchestrate data loading, training, testing, and saving."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        print_status("=" * 60)
        print_status("CIC-IoMT-2024 Multi-class Intrusion Detection Training (Feature Selection)")
        print_status("=" * 60)

        if self.args.test_size + self.args.val_size >= 1.0:
            raise ValueError("test_size + val_size must be < 1.0")

        print_status("Initializing data loader...")
        data_loader = DataLoaderPreprocessor(
            label_col=self.args.label_col,
            drop_cols=self.args.drop_cols,
            encode_categoricals=self.args.encode_categoricals,
        )

        train_df = None
        test_df = None

        print_status("Loading datasets...")
        if self.args.train_path:
            train_df = data_loader.load_parquet(Path(self.args.train_path), sample_size=self.args.sample_size)
            if self.args.test_path:
                test_df = data_loader.load_parquet(Path(self.args.test_path), sample_size=None)
        else:
            train_df = data_loader.load_parquet(Path(self.args.data_path), sample_size=self.args.sample_size)

        print_status("Preprocessing training data...")
        X_train_df, y_train_raw, feature_cols, categorical_cols, feature_stats, dropped_features = (
            data_loader.prepare(
                train_df,
                build_maps=True,
                null_zero_threshold=self.args.null_zero_threshold,
            )
        )

        del train_df
        import gc
        gc.collect()
        print_status("Freed training dataframe memory", level=1)

        if test_df is not None:
            print_status("Preprocessing test data...")
            X_test_df, y_test_raw, _, _, _, _ = data_loader.prepare(
                test_df, build_maps=False, null_zero_threshold=None
            )

            missing_cols = set(feature_cols) - set(X_test_df.columns)
            extra_cols = set(X_test_df.columns) - set(feature_cols)
            if missing_cols:
                print_status(
                    f"Adding {len(missing_cols)} missing columns to test set (filled with 0)",
                    level=1,
                )
            if extra_cols:
                print_status(f"Dropping {len(extra_cols)} extra columns from test set", level=1)

            X_test_df = X_test_df.reindex(columns=feature_cols, fill_value=0)

            del test_df
            gc.collect()
            print_status("Freed test dataframe memory", level=1)
        else:
            X_test_df = None
            y_test_raw = None

        print_status("Encoding labels...")
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw.astype(str))
        print_status(f"Encoded {len(label_encoder.classes_)} classes", level=1)
        print_status(
            f"Classes: {', '.join(label_encoder.classes_[:5])}..."
            if len(label_encoder.classes_) > 5
            else f"Classes: {', '.join(label_encoder.classes_)}",
            level=1,
        )

        if y_test_raw is not None:
            unseen_labels = set(y_test_raw.astype(str).unique()) - set(label_encoder.classes_)
            if unseen_labels:
                print_status(
                    f"Warning: {len(unseen_labels)} unseen labels in test set, will be encoded as -1",
                    level=1,
                )
                y_test_str = y_test_raw.astype(str)
                for unseen in unseen_labels:
                    y_test_str = y_test_str.replace(unseen, "__UNSEEN__")
                all_labels = list(label_encoder.classes_) + ["__UNSEEN__"]
                extended_encoder = LabelEncoder()
                extended_encoder.fit(all_labels)
                y_test = extended_encoder.transform(y_test_str)
            else:
                y_test = label_encoder.transform(y_test_raw.astype(str))
        else:
            y_test = None

        print_status("Splitting data into train/val/test sets...")
        if X_test_df is None:
            print_status(
                f"Splitting: train={1-self.args.test_size-self.args.val_size:.1%}, "
                f"val={self.args.val_size:.1%}, test={self.args.test_size:.1%}",
                level=1,
            )
            X_train, X_temp, y_train, y_temp = train_test_split(
                X_train_df,
                y_train,
                test_size=self.args.test_size + self.args.val_size,
                random_state=self.args.random_seed,
                stratify=y_train,
            )

            val_relative = self.args.val_size / (self.args.test_size + self.args.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=1 - val_relative,
                random_state=self.args.random_seed,
                stratify=y_temp,
            )
            print_status(
                f"Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}",
                level=1,
            )
        else:
            print_status(f"Splitting train into train/val: val={self.args.val_size:.1%}", level=1)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_df,
                y_train,
                test_size=self.args.val_size,
                random_state=self.args.random_seed,
                stratify=y_train,
            )
            X_test = X_test_df
            print_status(
                f"Split sizes: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}",
                level=1,
            )

        del X_train_df
        gc.collect()
        print_status("Freed preprocessing dataframe memory", level=1)

        trainer = ModelTrainer(
            n_estimators=self.args.n_estimators,
            max_depth=self.args.max_depth,
            num_leaves=self.args.num_leaves,
            learning_rate=self.args.learning_rate,
            subsample=self.args.subsample,
            colsample_bytree=self.args.colsample_bytree,
            min_data_in_leaf=self.args.min_data_in_leaf,
        )

        print_status("=" * 60)
        print_status("Training selector model for feature importance...")
        selector_model = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=len(label_encoder.classes_),
            class_weight=self.args.class_weight,
            early_stopping_rounds=self.args.early_stopping_rounds,
            categorical_feature=categorical_cols,
        )

        selected_features = select_top_features(
            selector_model, feature_cols=feature_cols, top_features=self.args.top_features
        )
        print_status(f"Selected top {len(selected_features)} features", level=1)
        print_status(f"Top features: {selected_features[:10]}...", level=2)

        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "feature_importance_top.png"
        save_feature_importance_plot(
            selector_model,
            feature_cols=feature_cols,
            output_path=plot_path,
            top_n=self.args.top_features,
        )

        X_train = X_train[selected_features]
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]

        print_status("=" * 60)
        print_status("Training final model on selected features...")
        model = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            num_classes=len(label_encoder.classes_),
            class_weight=self.args.class_weight,
            early_stopping_rounds=self.args.early_stopping_rounds,
            categorical_feature=[c for c in categorical_cols if c in selected_features],
        )

        print_status("=" * 60)
        tester = ModelTester()
        report = tester.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            labels=label_encoder.classes_.tolist(),
        )
        print_status("=" * 60)
        print("\n" + report)
        print_status("=" * 60)

        print_status("Saving model and artifacts...")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print_status(f"Output directory: {output_dir}", level=1)

        model_path = output_dir / "lgbm_model.txt"
        print_status(f"Saving model to: {model_path}", level=1)
        model.booster_.save_model(str(model_path))
        print_status("Model saved successfully", level=2)

        metadata = {
            "label_col": self.args.label_col,
            "feature_cols": selected_features,
            "initial_feature_cols": feature_cols,
            "filtered_features": dropped_features,
            "feature_filter_threshold": self.args.null_zero_threshold,
            "label_classes": label_encoder.classes_.tolist(),
            "drop_cols": self.args.drop_cols,
            "encode_categoricals": self.args.encode_categoricals,
            "test_size": self.args.test_size,
            "val_size": self.args.val_size,
            "top_features": self.args.top_features,
            "feature_stats": feature_stats,
            "hyperparams": {
                "n_estimators": self.args.n_estimators,
                "max_depth": self.args.max_depth,
                "num_leaves": self.args.num_leaves,
                "learning_rate": self.args.learning_rate,
                "subsample": self.args.subsample,
                "colsample_bytree": self.args.colsample_bytree,
                "min_data_in_leaf": self.args.min_data_in_leaf,
                "class_weight": self.args.class_weight,
                "early_stopping_rounds": self.args.early_stopping_rounds,
            },
        }
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        report_path = output_dir / "classification_report.txt"
        print_status(f"Saving classification report to: {report_path}", level=1)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print_status("Classification report saved", level=2)

        print_status("=" * 60)
        print_status("Training completed successfully!")
        print_status(f"Model saved to: {model_path}")
        print_status(f"Metadata saved to: {output_dir / 'metadata.json'}")
        print_status(f"Report saved to: {report_path}")
        print_status("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM on CIC-IoMT-2024 data with feature selection"
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
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        dest="colsample_bytree",
        help="Column sampling rate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: Sample N rows from training data for faster testing (default: use all data)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Class weight strategy: 'balanced' (default) or 'none'",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds (default: 50, set to 0 to disable)",
    )
    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=20,
        help="Minimum data in leaf (default: 20)",
    )
    parser.add_argument(
        "--null-zero-threshold",
        type=float,
        default=0.4,
        help="Drop features where null or zero rate >= threshold (default: 0.4)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of top features to keep based on importance (default: 20)",
    )
    args = parser.parse_args()

    if args.class_weight == "none":
        args.class_weight = None

    if not args.data_path and not args.train_path:
        raise SystemExit("Provide --data-path or --train-path")
    if args.data_path and args.train_path:
        raise SystemExit("Use either --data-path or --train-path, not both")
    if args.test_path and not args.train_path:
        raise SystemExit("--test-path requires --train-path")
    if args.top_features < 1:
        raise SystemExit("--top-features must be >= 1")

    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "LightGBM is not installed. Install with: pip install lightgbm"
        ) from exc

    try:
        pipeline = TrainingPipeline(args)
        pipeline.run()
    except KeyboardInterrupt:
        print_status("\nTraining interrupted by user", level=0)
        sys.exit(1)
    except Exception as e:
        print_status(f"\nERROR: Training failed with exception: {type(e).__name__}: {e}", level=0)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
