from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.ml.preprocessing_iomt import load_and_prepare_multiclass
from src.ml.train_iomt_gbdt_fs import print_status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiclass LightGBM on IoMT data using sklearn-style preprocessing"
    )
    parser.add_argument("--data-path", required=True, help="Parquet file or directory with IoMT data")
    parser.add_argument("--label-col", required=True, help="Multiclass label column name")
    parser.add_argument("--output-dir", required=True, help="Directory to save artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row subsample for quick runs")

    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)

    args = parser.parse_args()

    model_name = "lgbm_mc"
    outdir = Path(args.output_dir)

    print_status("=" * 60)
    print_status("IoMT Multiclass Intrusion Detection - LightGBM", level=0)
    print_status("=" * 60)

    X_train, X_test, y_train, y_test, feature_names, preproc, class_map = load_and_prepare_multiclass(
        data_path=args.data_path,
        label_col=args.label_col,
        test_size=args.test_size,
        random_state=args.random_seed,
        sample_size=args.sample_size,
        base_outdir=str(outdir),
        model_name=model_name,
    )

    num_classes = len(class_map)
    print_status(
        f"Training multiclass LGBM on {X_train.shape[0]:,} samples, "
        f"{X_train.shape[1]:,} features, {num_classes} classes",
        level=1,
    )

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        num_leaves=args.num_leaves,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=args.random_seed,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric=["multi_logloss"],
        callbacks=[lgb.log_evaluation(50)],
    )
    print_status(f"Training completed in {time.time() - t0:.2f}s", level=1)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    target_names = [class_map[i] for i in sorted(class_map.keys())]
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    model_dir = outdir / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.txt"
    model.booster_.save_model(str(model_path))

    try:
        import joblib

        joblib.dump(preproc, model_dir / "preprocessor.pkl")
    except Exception as e:  # pragma: no cover - best-effort persistence
        print_status(f"Warning: failed to save preprocessor with joblib ({e})", level=1)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_test_samples": int(len(y_test)),
        "n_train_samples": int(len(y_train)),
        "n_features": int(X_train.shape[1]),
        "class_map": class_map,
        "hyperparams": {
            "num_leaves": args.num_leaves,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    }
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print_status("Evaluation on test set:", level=1)
    print("\n" + report)
    print_status(f"Model saved to: {model_path}", level=1)
    print_status(f"Metrics saved to: {model_dir / 'metrics.json'}", level=1)


if __name__ == "__main__":
    main()

