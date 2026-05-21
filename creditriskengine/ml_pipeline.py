"""
ml_pipeline.py — Production ML Pipeline for CreditRiskEngine
=============================================================

Provides end-to-end feature engineering and ensemble model inference for
credit-risk scoring.  Designed for the Zimbabwe micro-lending domain and
supports CatBoost, LightGBM, and XGBoost fold models serialised as pickle
artefacts.

Classes
-------
FeatureEngineer
    Fits frequency encodings and medians on training data, then transforms
    raw loan-application rows into 60+ numeric / categorical features.

CreditRiskModel
    Loads pre-trained fold models from disk and produces weighted-ensemble
    default-probability predictions.

Constants
---------
PRODUCT_MAP : dict
    Human-readable names for the six product codes (0–5).

Author : Inclusion Algorithm Team
Version: 2.0.0
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRODUCT_MAP: Dict[int, str] = {
    0: "Personal",
    1: "SME",
    2: "Agriculture",
    3: "Salary_Based",
    4: "Asset_Finance",
    5: "Emergency",
}


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """Production feature-engineering pipeline matching the notebook's 60+
    derived features.

    Workflow
    --------
    1. Instantiate: ``fe = FeatureEngineer()``
    2. Fit on training data: ``fe.fit(train_df)``
    3. Transform any data: ``df_feat = fe.transform(df)``
    4. Persist: ``fe.save('feature_engineer.pkl')``
    5. Reload: ``fe = FeatureEngineer.load('feature_engineer.pkl')``

    The transformer is deterministic and stateless after fitting; the only
    state captured during *fit* is per-column frequency maps and median
    imputation values.
    """

    # Categorical columns expected in the raw data
    CAT_FEATURES: List[str] = [
        "product_code",
        "payment_frequency",
        "loan_purpose",
        "client_gender",
        "marital_status",
        "employment_sector",
        "collateral_type",
        "disbursement_channel",
        "province",
    ]

    def __init__(self) -> None:
        self.fitted: bool = False
        self.freq_maps: Dict[str, Dict[Any, float]] = {}
        self.medians: Dict[str, float] = {}
        self.cat_features: List[str] = list(self.CAT_FEATURES)

    # ---- fit / transform --------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Compute frequency encodings and median imputation values.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame (raw, before any feature engineering).

        Returns
        -------
        self
        """
        for col in self.cat_features:
            if col in df.columns:
                self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.medians[col] = float(df[col].median())

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature-engineering transformations.

        Parameters
        ----------
        df : pd.DataFrame
            Raw loan-application data.

        Returns
        -------
        pd.DataFrame
            Feature-enriched copy of *df* (original is not mutated).
        """
        df = df.copy()

        # -- 1. Date parsing --------------------------------------------------
        date_cols = ["date_approved", "date_disbursed", "first_payment_due", "maturity_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")

        if "client_dob" in df.columns:
            df["client_dob"] = pd.to_datetime(df["client_dob"], format="mixed", dayfirst=True, errors="coerce")
            # Fix century-shifted dates (e.g. 2085 should be 1985)
            future_mask = df["client_dob"] > pd.Timestamp("2010-01-01")
            df.loc[future_mask, "client_dob"] = df.loc[future_mask, "client_dob"] - pd.DateOffset(years=100)

        # -- 2. Client age ----------------------------------------------------
        if "client_dob" in df.columns and "date_disbursed" in df.columns:
            df["client_age"] = (df["date_disbursed"] - df["client_dob"]).dt.days / 365.25
        elif "client_dob" in df.columns:
            df["client_age"] = (pd.Timestamp.now() - df["client_dob"]).dt.days / 365.25

        if "client_age" in df.columns:
            df["age_bucket"] = pd.cut(
                df["client_age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=[0, 1, 2, 3, 4],
            ).astype(float)

        # -- 3. Temporal features ---------------------------------------------
        if all(c in df.columns for c in ["maturity_date", "date_disbursed"]):
            df["loan_duration_days"] = (df["maturity_date"] - df["date_disbursed"]).dt.days
        if all(c in df.columns for c in ["date_disbursed", "date_approved"]):
            df["days_approval_to_disburse"] = (df["date_disbursed"] - df["date_approved"]).dt.days
        if all(c in df.columns for c in ["first_payment_due", "date_disbursed"]):
            df["days_to_first_payment"] = (df["first_payment_due"] - df["date_disbursed"]).dt.days

        if "date_approved" in df.columns:
            df["approval_month"] = df["date_approved"].dt.month
            df["approval_quarter"] = df["date_approved"].dt.quarter
            df["approval_dayofweek"] = df["date_approved"].dt.dayofweek

        # -- 4. Financial features --------------------------------------------
        rate = df.get("annual_rate_pct", pd.Series([15.0] * len(df)))
        rate = rate.fillna(rate.median() if rate.median() == rate.median() else 15.0)
        monthly_rate = rate / 100 / 12
        n = df.get("term_months", pd.Series([12] * len(df)))
        amount = df.get("amount_usd", pd.Series([1000] * len(df)))

        # Amortisation formula
        with np.errstate(divide="ignore", invalid="ignore"):
            df["est_monthly_payment"] = (
                amount * (monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
            )
        low_rate = monthly_rate < 0.001
        df.loc[low_rate, "est_monthly_payment"] = amount[low_rate] / n[low_rate]
        df["est_monthly_payment"] = df["est_monthly_payment"].fillna(amount / n)

        # Debt-to-income ratio
        income = df.get("monthly_income_usd", pd.Series([500] * len(df))).copy()
        income = income.fillna(self.medians.get("monthly_income_usd", 500))
        df["dti_ratio"] = (df["est_monthly_payment"] / income.replace(0, np.nan)).clip(0, 5)

        # Total cost & ratios
        df["total_loan_cost"] = df["est_monthly_payment"] * n
        df["interest_to_principal"] = (df["total_loan_cost"] - amount) / amount.replace(0, np.nan)
        df["income_to_loan"] = income / amount.replace(0, np.nan)
        df["loan_to_annual_income"] = amount / (income * 12).replace(0, np.nan)
        df["amount_per_term_month"] = amount / n.replace(0, np.nan)

        # Log transforms
        df["log_amount"] = np.log1p(amount)
        df["log_income"] = np.log1p(income)
        df["log_rate"] = np.log1p(rate)

        # -- 5. Risk flags ----------------------------------------------------
        df["is_mfi_loan"] = (rate > 50).astype(int)
        df["is_high_rate"] = (rate > 100).astype(int)
        df["is_short_term"] = (n <= 3).astype(int)
        df["is_long_term"] = (n >= 24).astype(int)

        collateral = df.get("collateral_type", pd.Series(["None"] * len(df)))
        df["has_real_collateral"] = collateral.isin(["Vehicle", "Property"]).astype(int)
        df["has_any_collateral"] = (~collateral.isin(["None", np.nan])).astype(int)

        months_emp = df.get("months_at_employer", pd.Series([12] * len(df)))
        df["stable_employment"] = (months_emp > 24).astype(int)
        df["very_new_employee"] = (months_emp < 3).astype(int)

        obligations = df.get("existing_obligations", pd.Series([0] * len(df)))
        df["high_obligations"] = (obligations >= 3).astype(int)

        age = df.get("client_age", pd.Series([35] * len(df)))
        df["is_young"] = (age < 25).astype(int)
        df["high_dti"] = (df.get("dti_ratio", pd.Series([0] * len(df))) > 0.5).astype(int)

        # -- 6. Interactions --------------------------------------------------
        df["rate_x_term"] = rate * n
        df["amount_x_rate"] = amount * rate
        df["obligations_ratio"] = obligations / (obligations + 1)

        # -- 7. Missingness indicators ----------------------------------------
        for col in [
            "monthly_income_usd",
            "collateral_type",
            "months_at_employer",
            "employment_sector",
            "annual_rate_pct",
        ]:
            miss_col = f"miss_{col}"
            if col in df.columns:
                df[miss_col] = df[col].isna().astype(int)
            else:
                df[miss_col] = 0

        miss_cols = [c for c in df.columns if c.startswith("miss_")]
        df["total_missing"] = df[miss_cols].sum(axis=1)

        # -- 8. Ordinal / binary encodings ------------------------------------
        freq_map = {"Monthly": 2, "Bi-Weekly": 1, "Weekly": 0}
        if "payment_frequency" in df.columns:
            df["payment_freq_ord"] = df["payment_frequency"].map(freq_map).fillna(2)
        else:
            df["payment_freq_ord"] = 2

        if "client_gender" in df.columns:
            df["gender_enc"] = (df["client_gender"] == "Male").astype(int)
        else:
            df["gender_enc"] = 0

        # -- 9. Frequency encodings -------------------------------------------
        for col in self.cat_features:
            if col in df.columns and col in self.freq_maps:
                df[f"{col}_freq"] = df[col].map(self.freq_maps[col]).fillna(0)

        return df

    # ---- feature lists ----------------------------------------------------

    def get_numeric_features(self) -> List[str]:
        """Return the ordered list of numeric feature names (for LightGBM /
        XGBoost which do not handle categoricals natively)."""
        return [
            # Raw numeric columns
            "amount_usd", "annual_rate_pct", "term_months", "monthly_income_usd",
            "existing_obligations", "num_dependents", "months_at_employer",
            # Derived
            "client_age", "age_bucket",
            "loan_duration_days", "days_approval_to_disburse", "days_to_first_payment",
            "approval_month", "approval_quarter", "approval_dayofweek",
            "est_monthly_payment", "dti_ratio", "total_loan_cost",
            "interest_to_principal", "income_to_loan", "loan_to_annual_income",
            "amount_per_term_month",
            "log_amount", "log_income", "log_rate",
            "is_mfi_loan", "is_high_rate", "is_short_term", "is_long_term",
            "has_real_collateral", "has_any_collateral",
            "stable_employment", "very_new_employee",
            "high_obligations", "is_young", "high_dti",
            "rate_x_term", "amount_x_rate", "obligations_ratio",
            "miss_monthly_income_usd", "miss_collateral_type", "miss_months_at_employer",
            "miss_employment_sector", "miss_annual_rate_pct", "total_missing",
            "payment_freq_ord", "gender_enc",
        ] + [f"{col}_freq" for col in self.cat_features]

    def get_catboost_features(self) -> List[str]:
        """Return numeric + categorical feature names for CatBoost."""
        return self.get_numeric_features() + self.cat_features

    # ---- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted FeatureEngineer to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        """Deserialise a FeatureEngineer from *path*."""
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ---------------------------------------------------------------------------
# Credit Risk Model (Ensemble Inference)
# ---------------------------------------------------------------------------

class CreditRiskModel:
    """Weighted ensemble of CatBoost + LightGBM + XGBoost fold models.

    Usage
    -----
    >>> model = CreditRiskModel(models_dir='models')
    >>> model.load()
    True
    >>> result = model.predict_single({'amount_usd': 5000, ...})
    """

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir: str = models_dir
        self.catboost_models: list = []
        self.lgb_models: list = []
        self.xgb_models: list = []
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.weights: Dict[str, float] = {"catboost": 0.6, "lgb": 0.2, "xgb": 0.2}
        self.feature_importance: Optional[pd.DataFrame] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_loaded: bool = False

    # ---- loading -----------------------------------------------------------

    def load(self) -> bool:
        """Load pre-trained fold models and the fitted FeatureEngineer.

        Returns
        -------
        bool
            ``True`` if loading succeeded, ``False`` otherwise.
        """
        try:
            fe_path = os.path.join(self.models_dir, "feature_engineer.pkl")
            self.feature_engineer = FeatureEngineer.load(fe_path)

            for artefact, attr in [
                ("catboost_models.pkl", "catboost_models"),
                ("lgb_models.pkl", "lgb_models"),
                ("xgb_models.pkl", "xgb_models"),
            ]:
                fpath = os.path.join(self.models_dir, artefact)
                if os.path.exists(fpath):
                    with open(fpath, "rb") as fh:
                        setattr(self, attr, pickle.load(fh))

            meta_path = os.path.join(self.models_dir, "model_metadata.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as fh:
                    meta = pickle.load(fh)
                    self.weights = meta.get("weights", self.weights)
                    self.feature_importance = meta.get("feature_importance")
                    self.training_metrics = meta.get("training_metrics", {})

            self.is_loaded = True
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"Error loading models: {exc}")
            return False

    # ---- prediction --------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row default probabilities for *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Raw loan-application data (before feature engineering).

        Returns
        -------
        np.ndarray
            1-D array of predicted default probabilities in [0, 1].

        Raises
        ------
        ValueError
            If models have not been loaded yet.
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load() first.")

        df_feat = self.feature_engineer.transform(df)

        preds = np.zeros(len(df_feat))
        total_weight = 0.0

        # -- CatBoost ---------------------------------------------------------
        if self.catboost_models:
            cb_features = [f for f in self.feature_engineer.get_catboost_features() if f in df_feat.columns]
            X_cb = df_feat[cb_features].copy()
            for col in self.feature_engineer.cat_features:
                if col in X_cb.columns:
                    X_cb[col] = X_cb[col].astype(str)
            cb_pred = np.mean([m.predict_proba(X_cb)[:, 1] for m in self.catboost_models], axis=0)
            preds += self.weights["catboost"] * cb_pred
            total_weight += self.weights["catboost"]

        # -- LightGBM --------------------------------------------------------
        if self.lgb_models:
            num_features = [f for f in self.feature_engineer.get_numeric_features() if f in df_feat.columns]
            X_num = df_feat[num_features].copy()
            lgb_pred = np.mean([m.predict_proba(X_num)[:, 1] for m in self.lgb_models], axis=0)
            preds += self.weights["lgb"] * lgb_pred
            total_weight += self.weights["lgb"]

        # -- XGBoost ----------------------------------------------------------
        if self.xgb_models:
            num_features = [f for f in self.feature_engineer.get_numeric_features() if f in df_feat.columns]
            X_num = df_feat[num_features].copy()
            xgb_pred = np.mean([m.predict_proba(X_num)[:, 1] for m in self.xgb_models], axis=0)
            preds += self.weights["xgb"] * xgb_pred
            total_weight += self.weights["xgb"]

        if total_weight > 0:
            preds /= total_weight

        return preds

    def predict_single(self, loan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict default risk for a single loan application.

        Parameters
        ----------
        loan_data : dict
            Raw field values for one loan application.

        Returns
        -------
        dict
            Keys: ``probability_of_default``, ``risk_level``, ``risk_color``,
            ``risk_score`` (0–1000), ``recommendation`` (APPROVE / REVIEW /
            DECLINE).
        """
        df = pd.DataFrame([loan_data])
        prob = float(self.predict_proba(df)[0])

        # Risk classification bands
        if prob < 0.10:
            risk_level, risk_color = "Very Low", "#00C853"
        elif prob < 0.25:
            risk_level, risk_color = "Low", "#69F0AE"
        elif prob < 0.40:
            risk_level, risk_color = "Medium", "#FFD600"
        elif prob < 0.60:
            risk_level, risk_color = "High", "#FF6D00"
        else:
            risk_level, risk_color = "Very High", "#FF1744"

        if prob < 0.35:
            recommendation = "APPROVE"
        elif prob < 0.50:
            recommendation = "REVIEW"
        else:
            recommendation = "DECLINE"

        return {
            "probability_of_default": round(prob, 4),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_score": round((1 - prob) * 1000),
            "recommendation": recommendation,
        }
