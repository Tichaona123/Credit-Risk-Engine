"""
ifrs9_engine.py — IFRS 9 Staging & Expected Credit Loss Engine
===============================================================

Implements the three-stage impairment model mandated by IFRS 9 *Financial
Instruments* for the CreditRiskEngine platform.  Supports:

* **Stage assignment** (1 / 2 / 3) based on 12-month PD and days-past-due.
* **ECL calculation** — 12-month ECL for Stage 1, lifetime ECL for Stages 2/3.
* **Portfolio-level staging** with aggregated exposure and provision metrics.
* **Forward-looking adjustments** using probability-weighted macroeconomic
  scenarios (optimistic / baseline / pessimistic).

All monetary values are denominated in USD.

Author : Inclusion Algorithm Team
Version: 2.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IFRS9Config:
    """Tuneable parameters for the IFRS 9 engine.

    Attributes
    ----------
    lgd_unsecured : float
        Loss-Given-Default for unsecured loans (default 40 %).
    lgd_secured : float
        Loss-Given-Default for secured / collateralised loans (default 25 %).
    discount_rate : float
        Effective interest rate used to discount lifetime ECL (default 15 %).
    stage1_threshold : float
        PD threshold above which a loan moves from Stage 1 → Stage 2.
    stage2_threshold : float
        PD threshold above which a loan is classified as Stage 3.
    fw_optimistic_weight : float
        Probability weight for the *optimistic* macro-economic scenario.
    fw_baseline_weight : float
        Probability weight for the *baseline* macro-economic scenario.
    fw_pessimistic_weight : float
        Probability weight for the *pessimistic* macro-economic scenario.
    """

    lgd_unsecured: float = 0.40
    lgd_secured: float = 0.25
    discount_rate: float = 0.15
    stage1_threshold: float = 0.15
    stage2_threshold: float = 0.40
    # Forward-looking scenario weights (must sum to 1.0)
    fw_optimistic_weight: float = 0.20
    fw_baseline_weight: float = 0.50
    fw_pessimistic_weight: float = 0.30


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IFRS9Engine:
    """Core IFRS 9 staging and ECL calculation engine.

    Parameters
    ----------
    config : IFRS9Config, optional
        Override default thresholds and LGD assumptions.  If ``None`` the
        engine uses ``IFRS9Config()`` defaults.

    Examples
    --------
    >>> engine = IFRS9Engine()
    >>> engine.assign_stage(pd_12m=0.08)
    1
    >>> ecl = engine.calculate_ecl(amount=10_000, pd_12m=0.08, stage=1)
    >>> ecl['ecl']
    320.0
    """

    # Human-readable stage labels (index 0 unused)
    _STAGE_NAMES = ["", "Performing", "Underperforming", "Non-Performing"]

    def __init__(self, config: Optional[IFRS9Config] = None) -> None:
        self.config = config or IFRS9Config()

    # ---- stage assignment --------------------------------------------------

    def assign_stage(self, pd_12m: float, days_past_due: int = 0) -> int:
        """Assign an IFRS 9 impairment stage.

        Parameters
        ----------
        pd_12m : float
            12-month probability of default (0–1).
        days_past_due : int
            Number of days the borrower is past due on payments.

        Returns
        -------
        int
            Stage number: 1 (Performing), 2 (Under-performing), or
            3 (Non-performing / credit-impaired).
        """
        if days_past_due > 90 or pd_12m >= self.config.stage2_threshold:
            return 3
        if pd_12m >= self.config.stage1_threshold:
            return 2
        return 1

    # ---- single-loan ECL ---------------------------------------------------

    def calculate_ecl(
        self,
        amount: float,
        pd_12m: float,
        stage: int,
        term_months: int = 12,
        has_collateral: bool = False,
    ) -> Dict[str, Any]:
        """Calculate Expected Credit Loss for a single exposure.

        Stage 1 uses a 12-month horizon; Stages 2 and 3 use a lifetime
        horizon discounted at the effective interest rate.

        Parameters
        ----------
        amount : float
            Exposure at Default (EAD) in USD.
        pd_12m : float
            12-month probability of default.
        stage : int
            IFRS 9 stage (1, 2, or 3).
        term_months : int
            Remaining contractual term in months.
        has_collateral : bool
            Whether the loan is backed by recognised collateral.

        Returns
        -------
        dict
            ``stage``, ``stage_name``, ``pd_12m``, ``lgd``, ``ead``,
            ``ecl``, ``ecl_rate`` (%), ``provision_category``.
        """
        lgd = self.config.lgd_secured if has_collateral else self.config.lgd_unsecured

        if stage == 1:
            # 12-month ECL
            ecl = amount * pd_12m * lgd
        else:
            # Lifetime ECL — convert annual PD to cumulative PD over the
            # remaining term, then present-value-discount the loss.
            years = term_months / 12.0
            lifetime_pd = 1 - (1 - pd_12m) ** years
            discount = 1 / (1 + self.config.discount_rate) ** (years / 2)
            ecl = amount * lifetime_pd * lgd * discount

        return {
            "stage": stage,
            "stage_name": self._STAGE_NAMES[stage],
            "pd_12m": round(pd_12m, 4),
            "lgd": lgd,
            "ead": amount,
            "ecl": round(ecl, 2),
            "ecl_rate": round(ecl / amount * 100, 2) if amount > 0 else 0.0,
            "provision_category": "Specific" if stage == 3 else "General",
        }

    # ---- portfolio-level staging -------------------------------------------

    def portfolio_staging(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute portfolio-level staging distribution and total ECL.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``amount_usd`` and ``pd_12m`` columns.
            Optional: ``has_collateral`` (bool), ``term_months`` (int).

        Returns
        -------
        dict
            Per-stage counts, exposures, average PD, and ECL totals, plus
            ``total_ecl``, ``total_exposure``, ``coverage_ratio``, and an
            enriched ``df`` with ``stage`` and ``ecl`` columns appended.
        """
        df = df.copy()
        df["stage"] = df["pd_12m"].apply(lambda x: self.assign_stage(x))

        # Per-stage summaries
        results: Dict[str, Any] = {}
        for stage in (1, 2, 3):
            mask = df["stage"] == stage
            stage_df = df[mask]
            results[f"stage_{stage}"] = {
                "count": int(mask.sum()),
                "pct": round(float(mask.mean()) * 100, 1),
                "total_exposure": round(float(stage_df["amount_usd"].sum()), 2),
                "avg_pd": round(float(stage_df["pd_12m"].mean()), 4) if len(stage_df) > 0 else 0.0,
                "total_ecl": 0.0,  # populated below
            }

        # Row-level ECL
        ecls: List[float] = []
        for _, row in df.iterrows():
            has_coll = bool(row.get("has_collateral", False))
            term = int(row.get("term_months", 12))
            ecl_result = self.calculate_ecl(
                amount=float(row["amount_usd"]),
                pd_12m=float(row["pd_12m"]),
                stage=int(row["stage"]),
                term_months=term,
                has_collateral=has_coll,
            )
            ecls.append(ecl_result["ecl"])

        df["ecl"] = ecls

        for stage in (1, 2, 3):
            mask = df["stage"] == stage
            results[f"stage_{stage}"]["total_ecl"] = round(float(df.loc[mask, "ecl"].sum()), 2)

        total_exposure = float(df["amount_usd"].sum())
        total_ecl = float(df["ecl"].sum())

        results["total_ecl"] = round(total_ecl, 2)
        results["total_exposure"] = round(total_exposure, 2)
        results["coverage_ratio"] = round(total_ecl / total_exposure * 100, 2) if total_exposure > 0 else 0.0
        results["df"] = df  # enriched DataFrame

        return results

    # ---- forward-looking adjustments ---------------------------------------

    def forward_looking_adjustment(
        self,
        base_ecl: float,
        macro_scenario: str = "baseline",  # noqa: ARG002 — kept for API symmetry
    ) -> Dict[str, Any]:
        """Apply probability-weighted forward-looking overlays per IFRS 9.

        Three scenarios are combined using the weights stored in
        :class:`IFRS9Config`:

        * **Optimistic** — multiplier 0.85
        * **Baseline** — multiplier 1.00
        * **Pessimistic** — multiplier 1.25

        Parameters
        ----------
        base_ecl : float
            Unadjusted ECL amount.
        macro_scenario : str
            Informational label (not used in calculation; the engine always
            probability-weights all three scenarios).

        Returns
        -------
        dict
            ``base_ecl``, ``weighted_ecl``, ``adjustment_pct``, and per-
            scenario breakdown.
        """
        multipliers = {
            "optimistic": 0.85,
            "baseline": 1.00,
            "pessimistic": 1.25,
        }

        weighted_ecl = (
            base_ecl * multipliers["optimistic"] * self.config.fw_optimistic_weight
            + base_ecl * multipliers["baseline"] * self.config.fw_baseline_weight
            + base_ecl * multipliers["pessimistic"] * self.config.fw_pessimistic_weight
        )

        return {
            "base_ecl": round(base_ecl, 2),
            "weighted_ecl": round(weighted_ecl, 2),
            "adjustment_pct": round((weighted_ecl / base_ecl - 1) * 100, 2) if base_ecl > 0 else 0.0,
            "scenarios": {
                name: round(base_ecl * mult, 2) for name, mult in multipliers.items()
            },
        }
