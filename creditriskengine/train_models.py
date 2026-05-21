"""
train_models.py — Model Training Script for CreditRiskEngine
=============================================================

End-to-end training pipeline that:

1. Loads ``Train.csv`` from a configurable data directory.
2. Fits the :class:`FeatureEngineer` on training data.
3. Runs CatBoost hyper-parameter optimisation via Optuna (20 trials default,
   5 in ``--quick`` mode).
4. Trains LightGBM and XGBoost with stratified 5-fold cross-validation.
5. Performs multi-seed CatBoost averaging (3 seeds default, 2 in quick mode).
6. Persists all fold models, the fitted feature engineer, and training
   metadata to the ``models/`` directory.
7. Prints a summary of training metrics (AUC, log-loss, Gini).

Usage
-----
::

    # Full training run
    python train_models.py --data-dir ../../

    # Quick demo (fewer Optuna trials & seeds)
    python train_models.py --data-dir ../../ --quick

Author : Inclusion Algorithm Team
Version: 2.0.0
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Lazy imports so the script can display a helpful error when libs are missing.
_MISSING_LIBS: List[str] = []

try:
    from catboost import CatBoostClassifier, Pool
except ImportError:
    _MISSING_LIBS.append("catboost")

try:
    import lightgbm as lgb
except ImportError:
    _MISSING_LIBS.append("lightgbm")

try:
    import xgboost as xgb
except ImportError:
    _MISSING_LIBS.append("xgboost")

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    _MISSING_LIBS.append("optuna")

# Local imports
from ml_pipeline import FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_deps() -> None:
    """Abort early with an actionable message if dependencies are missing."""
    if _MISSING_LIBS:
        print(
            "ERROR — the following required packages are not installed:\n  "
            + ", ".join(_MISSING_LIBS)
            + "\n\nInstall them with:\n  pip install "
            + " ".join(_MISSING_LIBS)
        )
        sys.exit(1)


def _gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised Gini coefficient (2 × AUC − 1)."""
    return 2 * roc_auc_score(y_true, y_pred) - 1


def _print_banner(text: str) -> None:
    width = max(len(text) + 4, 60)
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Optuna objective for CatBoost
# ---------------------------------------------------------------------------

def _catboost_objective(
    trial: "optuna.Trial",
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: List[str],
    n_folds: int = 5,
) -> float:
    """Optuna objective that returns mean CV log-loss for CatBoost."""
    params = {
        "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
        "depth": trial.suggest_int("depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.5, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        "eval_metric": "Logloss",
        "loss_function": "Logloss",
        "verbose": 0,
        "random_seed": 42,
        "auto_class_weights": "Balanced",
        "cat_features": cat_features,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores: List[float] = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=(Xval, yval),
            early_stopping_rounds=50,
            verbose=0,
        )
        preds = model.predict_proba(Xval)[:, 1]
        scores.append(log_loss(yval, preds))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def train_catboost_multiseed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: Dict[str, Any],
    cat_features: List[str],
    seeds: List[int],
    n_folds: int = 5,
) -> Tuple[List[Any], Dict[str, float]]:
    """Train CatBoost across multiple seeds, returning fold models and metrics."""
    all_models: List[Any] = []
    oof_preds = np.zeros(len(X_train))
    oof_counts = np.zeros(len(X_train))

    for seed in seeds:
        params = {**best_params, "random_seed": seed, "verbose": 0,
                  "auto_class_weights": "Balanced", "cat_features": cat_features}

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
            ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(Xtr, ytr, eval_set=(Xval, yval), early_stopping_rounds=50, verbose=0)
            all_models.append(model)

            preds = model.predict_proba(Xval)[:, 1]
            oof_preds[val_idx] += preds
            oof_counts[val_idx] += 1

            auc = roc_auc_score(yval, preds)
            print(f"    Seed {seed} | Fold {fold} — AUC: {auc:.4f}")

    # Average OOF predictions across seeds
    mask = oof_counts > 0
    oof_preds[mask] /= oof_counts[mask]

    metrics = {
        "auc": round(roc_auc_score(y_train[mask], oof_preds[mask]), 4),
        "gini": round(_gini(y_train[mask], oof_preds[mask]), 4),
        "logloss": round(log_loss(y_train[mask], oof_preds[mask]), 4),
    }
    return all_models, metrics


def train_lgb_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[List[Any], Dict[str, float]]:
    """Train LightGBM with stratified K-fold CV."""
    params = {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "verbose": -1,
        "is_unbalance": True,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models: List[Any] = []
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(model)

        preds = model.predict_proba(Xval)[:, 1]
        oof_preds[val_idx] = preds
        auc = roc_auc_score(yval, preds)
        print(f"    Fold {fold} — AUC: {auc:.4f}")

    metrics = {
        "auc": round(roc_auc_score(y_train, oof_preds), 4),
        "gini": round(_gini(y_train, oof_preds), 4),
        "logloss": round(log_loss(y_train, oof_preds), 4),
    }
    return models, metrics


def train_xgb_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int = 5,
    seed: int = 42,
) -> Tuple[List[Any], Dict[str, float]]:
    """Train XGBoost with stratified K-fold CV."""
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    params = {
        "n_estimators": 800,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "random_state": seed,
        "verbosity": 0,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models: List[Any] = []
    oof_preds = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            Xtr, ytr,
            eval_set=[(Xval, yval)],
            verbose=False,
        )
        models.append(model)

        preds = model.predict_proba(Xval)[:, 1]
        oof_preds[val_idx] = preds
        auc = roc_auc_score(yval, preds)
        print(f"    Fold {fold} — AUC: {auc:.4f}")

    metrics = {
        "auc": round(roc_auc_score(y_train, oof_preds), 4),
        "gini": round(_gini(y_train, oof_preds), 4),
        "logloss": round(log_loss(y_train, oof_preds), 4),
    }
    return models, metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the full training pipeline."""
    parser = argparse.ArgumentParser(
        description="CreditRiskEngine — Train ensemble credit-risk models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python train_models.py --data-dir ../../\n  python train_models.py --data-dir ../../ --quick",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../../",
        help="Directory containing Train.csv (default: ../../)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Output directory for saved models (default: models/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer Optuna trials (5) and seeds (2) for fast demo",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )

    args = parser.parse_args()

    # Check dependencies before doing anything expensive
    _check_deps()

    n_optuna_trials = 5 if args.quick else 20
    seeds = [42, 123] if args.quick else [42, 123, 2024]

    _print_banner("CreditRiskEngine — Model Training Pipeline")
    print(f"  Mode         : {'QUICK (demo)' if args.quick else 'FULL'}")
    print(f"  Data dir     : {os.path.abspath(args.data_dir)}")
    print(f"  Models dir   : {os.path.abspath(args.models_dir)}")
    print(f"  Optuna trials: {n_optuna_trials}")
    print(f"  Seeds        : {seeds}")
    print(f"  CV folds     : {args.n_folds}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    _print_banner("Step 1 / 6 — Loading Data")

    csv_path = os.path.join(args.data_dir, "Train.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: Train.csv not found at {os.path.abspath(csv_path)}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns from {csv_path}")
    print(f"  Target distribution:\n{df['Target'].value_counts().to_string()}")
    print(f"  Default rate: {df['Target'].mean():.2%}")

    # ------------------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------------------
    _print_banner("Step 2 / 6 — Feature Engineering")

    fe = FeatureEngineer()
    fe.fit(df)
    df_feat = fe.transform(df)

    print(f"  Derived features: {len(df_feat.columns)} total columns")

    # Prepare feature matrices
    y = df["Target"]

    # CatBoost features
    cb_features = [f for f in fe.get_catboost_features() if f in df_feat.columns]
    X_cb = df_feat[cb_features].copy()
    cat_cols_in_X = [c for c in fe.cat_features if c in X_cb.columns]
    for col in cat_cols_in_X:
        X_cb[col] = X_cb[col].astype(str)

    # Numeric features (LGB + XGB)
    num_features = [f for f in fe.get_numeric_features() if f in df_feat.columns]
    X_num = df_feat[num_features].copy()

    print(f"  CatBoost feature count : {len(cb_features)}")
    print(f"  LGB / XGB feature count: {len(num_features)}")

    # ------------------------------------------------------------------
    # 3. CatBoost HPO with Optuna
    # ------------------------------------------------------------------
    _print_banner("Step 3 / 6 — CatBoost Hyper-Parameter Optimisation (Optuna)")

    t0 = time.time()
    study = optuna.create_study(direction="minimize", study_name="catboost_hpo")
    study.optimize(
        lambda trial: _catboost_objective(trial, X_cb, y, cat_cols_in_X, n_folds=args.n_folds),
        n_trials=n_optuna_trials,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    best_params = study.best_params
    # Ensure loss / eval metric are present
    best_params["loss_function"] = "Logloss"
    best_params["eval_metric"] = "Logloss"

    print(f"\n  Best trial #{study.best_trial.number} — LogLoss: {study.best_value:.5f}")
    print(f"  Optimisation took {elapsed:.1f}s")
    print("  Best params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    # ------------------------------------------------------------------
    # 4. CatBoost multi-seed training
    # ------------------------------------------------------------------
    _print_banner("Step 4 / 6 — CatBoost Multi-Seed Training")

    cb_models, cb_metrics = train_catboost_multiseed(
        X_cb, y, best_params, cat_cols_in_X, seeds, n_folds=args.n_folds,
    )
    print(f"\n  CatBoost OOF — AUC: {cb_metrics['auc']:.4f}  Gini: {cb_metrics['gini']:.4f}  LogLoss: {cb_metrics['logloss']:.4f}")

    # ------------------------------------------------------------------
    # 5. LightGBM + XGBoost CV
    # ------------------------------------------------------------------
    _print_banner("Step 5 / 6 — LightGBM 5-Fold CV")

    lgb_models, lgb_metrics = train_lgb_cv(X_num, y, n_folds=args.n_folds)
    print(f"\n  LightGBM OOF — AUC: {lgb_metrics['auc']:.4f}  Gini: {lgb_metrics['gini']:.4f}  LogLoss: {lgb_metrics['logloss']:.4f}")

    _print_banner("Step 5b / 6 — XGBoost 5-Fold CV")

    xgb_models, xgb_metrics = train_xgb_cv(X_num, y, n_folds=args.n_folds)
    print(f"\n  XGBoost OOF  — AUC: {xgb_metrics['auc']:.4f}  Gini: {xgb_metrics['gini']:.4f}  LogLoss: {xgb_metrics['logloss']:.4f}")

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    _print_banner("Step 6 / 6 — Saving Models & Metadata")

    models_dir = args.models_dir
    os.makedirs(models_dir, exist_ok=True)

    # Feature engineer
    fe_path = os.path.join(models_dir, "feature_engineer.pkl")
    fe.save(fe_path)
    print(f"  ✔ Feature engineer → {fe_path}")

    # CatBoost
    cb_path = os.path.join(models_dir, "catboost_models.pkl")
    with open(cb_path, "wb") as fh:
        pickle.dump(cb_models, fh)
    print(f"  ✔ CatBoost models ({len(cb_models)}) → {cb_path}")

    # LightGBM
    lgb_path = os.path.join(models_dir, "lgb_models.pkl")
    with open(lgb_path, "wb") as fh:
        pickle.dump(lgb_models, fh)
    print(f"  ✔ LightGBM models ({len(lgb_models)}) → {lgb_path}")

    # XGBoost
    xgb_path = os.path.join(models_dir, "xgb_models.pkl")
    with open(xgb_path, "wb") as fh:
        pickle.dump(xgb_models, fh)
    print(f"  ✔ XGBoost models ({len(xgb_models)}) → {xgb_path}")

    # Feature importance (from first CatBoost model)
    try:
        fi = cb_models[0].get_feature_importance(prettified=True)
        fi_df = pd.DataFrame(fi).rename(columns={"Feature Id": "feature", "Importances": "importance"})
    except Exception:
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    # Metadata
    meta = {
        "weights": {"catboost": 0.6, "lgb": 0.2, "xgb": 0.2},
        "feature_importance": fi_df,
        "training_metrics": {
            "catboost": cb_metrics,
            "lgb": lgb_metrics,
            "xgb": xgb_metrics,
        },
        "n_training_rows": len(df),
        "n_features_catboost": len(cb_features),
        "n_features_numeric": len(num_features),
        "optuna_best_params": best_params,
        "seeds": seeds,
    }
    meta_path = os.path.join(models_dir, "model_metadata.pkl")
    with open(meta_path, "wb") as fh:
        pickle.dump(meta, fh)
    print(f"  ✔ Metadata → {meta_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_banner("Training Complete — Summary")
    print(f"  {'Model':<12} {'AUC':>8} {'Gini':>8} {'LogLoss':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
    for name, m in [("CatBoost", cb_metrics), ("LightGBM", lgb_metrics), ("XGBoost", xgb_metrics)]:
        print(f"  {name:<12} {m['auc']:>8.4f} {m['gini']:>8.4f} {m['logloss']:>10.4f}")
    print(f"\n  All artefacts saved to: {os.path.abspath(models_dir)}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
