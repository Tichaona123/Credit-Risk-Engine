"""
stress_testing.py — Macro-Economic Stress-Testing Framework
===========================================================

Provides deterministic scenario-based stress testing and PD × LGD sensitivity
analysis for a credit-risk portfolio.

Classes
-------
StressScenario
    Dataclass describing a single macro-economic scenario (PD multiplier,
    LGD multiplier, portfolio growth).

StressTestEngine
    Runs all configured scenarios against a portfolio DataFrame and produces
    a sensitivity heatmap.

Constants
---------
DEFAULT_SCENARIOS : dict[str, StressScenario]
    Four pre-defined scenarios — Baseline, Adverse, Severe, Extreme — that
    cover the continuum from expected conditions to a full financial crisis
    with currency collapse (relevant for Zimbabwe's macro environment).

Author : Inclusion Algorithm Team
Version: 2.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A single macro-economic stress scenario.

    Attributes
    ----------
    name : str
        Short human-readable label.
    description : str
        Longer narrative of the economic conditions assumed.
    pd_multiplier : float
        Factor applied to each loan's 12-month PD.
    lgd_multiplier : float
        Factor applied to the base LGD.
    exposure_growth : float
        Assumed portfolio growth (negative = shrinkage).
    """

    name: str
    description: str
    pd_multiplier: float
    lgd_multiplier: float
    exposure_growth: float = 0.0


DEFAULT_SCENARIOS: Dict[str, StressScenario] = {
    "Baseline": StressScenario(
        name="Baseline",
        description="Expected economic conditions",
        pd_multiplier=1.0,
        lgd_multiplier=1.0,
        exposure_growth=0.0,
    ),
    "Adverse": StressScenario(
        name="Adverse",
        description="Mild recession — unemployment +3 %, GDP −2 %",
        pd_multiplier=1.5,
        lgd_multiplier=1.1,
        exposure_growth=-0.05,
    ),
    "Severe": StressScenario(
        name="Severe",
        description="Severe recession — unemployment +6 %, GDP −5 %",
        pd_multiplier=2.5,
        lgd_multiplier=1.25,
        exposure_growth=-0.10,
    ),
    "Extreme": StressScenario(
        name="Extreme",
        description="Financial crisis — unemployment +10 %, GDP −8 %, currency collapse",
        pd_multiplier=4.0,
        lgd_multiplier=1.5,
        exposure_growth=-0.20,
    ),
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class StressTestEngine:
    """Run scenario-based stress tests and sensitivity analysis on a credit
    portfolio.

    Parameters
    ----------
    scenarios : dict[str, StressScenario], optional
        Custom scenario map.  Defaults to :data:`DEFAULT_SCENARIOS`.

    Examples
    --------
    >>> engine = StressTestEngine()
    >>> results = engine.run_stress_test(portfolio_df, lgd_base=0.40)
    >>> results['Severe']['ecl_increase_pct']
    185.3
    """

    def __init__(self, scenarios: Optional[Dict[str, StressScenario]] = None) -> None:
        self.scenarios: Dict[str, StressScenario] = scenarios or DEFAULT_SCENARIOS

    # ---- core stress test --------------------------------------------------

    def run_stress_test(
        self,
        portfolio_df: pd.DataFrame,
        lgd_base: float = 0.40,
    ) -> Dict[str, Dict[str, Any]]:
        """Run every configured scenario against the portfolio.

        Parameters
        ----------
        portfolio_df : pd.DataFrame
            Must contain ``amount_usd`` and ``pd_12m`` columns.
        lgd_base : float
            Base Loss-Given-Default before stress adjustment.

        Returns
        -------
        dict[str, dict]
            Per-scenario results keyed by scenario name.  Each entry
            contains stressed PD, ECL, exposure, capital-impact figures, etc.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for name, scenario in self.scenarios.items():
            stressed_pd = np.clip(
                portfolio_df["pd_12m"] * scenario.pd_multiplier, 0, 1
            )
            stressed_lgd = min(lgd_base * scenario.lgd_multiplier, 1.0)
            stressed_exposure = portfolio_df["amount_usd"] * (1 + scenario.exposure_growth)

            stressed_ecl = float((stressed_exposure * stressed_pd * stressed_lgd).sum())
            base_ecl = float((portfolio_df["amount_usd"] * portfolio_df["pd_12m"] * lgd_base).sum())
            stressed_default_rate = float(stressed_pd.mean())

            results[name] = {
                "scenario": name,
                "description": scenario.description,
                "pd_multiplier": scenario.pd_multiplier,
                "lgd_multiplier": scenario.lgd_multiplier,
                "avg_stressed_pd": round(stressed_default_rate, 4),
                "stressed_ecl": round(stressed_ecl, 2),
                "base_ecl": round(base_ecl, 2),
                "ecl_increase_pct": (
                    round((stressed_ecl / base_ecl - 1) * 100, 1) if base_ecl > 0 else 0.0
                ),
                "total_exposure": round(float(stressed_exposure.sum()), 2),
                "capital_impact": round(stressed_ecl - base_ecl, 2),
            }

        return results

    # ---- sensitivity matrix ------------------------------------------------

    def sensitivity_analysis(
        self,
        portfolio_df: pd.DataFrame,
        lgd_base: float = 0.40,
        pd_range: Tuple[float, float, int] = (0.5, 4.0, 8),
        lgd_range: Tuple[float, float, int] = (0.8, 1.5, 8),
    ) -> pd.DataFrame:
        """Generate a PD-multiplier × LGD-multiplier sensitivity matrix.

        Parameters
        ----------
        portfolio_df : pd.DataFrame
            Must contain ``amount_usd`` and ``pd_12m``.
        lgd_base : float
            Base LGD assumption.
        pd_range : tuple(float, float, int)
            ``(start, stop, num_steps)`` for PD multipliers.
        lgd_range : tuple(float, float, int)
            ``(start, stop, num_steps)`` for LGD multipliers.

        Returns
        -------
        pd.DataFrame
            Heatmap-ready DataFrame with PD multipliers as rows and LGD
            multipliers as columns; cell values are total stressed ECL.
        """
        pd_multipliers = np.linspace(*pd_range)
        lgd_multipliers = np.linspace(*lgd_range)

        matrix: List[List[float]] = []
        for pd_m in pd_multipliers:
            row: List[float] = []
            for lgd_m in lgd_multipliers:
                stressed_ecl = float(
                    (
                        portfolio_df["amount_usd"]
                        * np.clip(portfolio_df["pd_12m"] * pd_m, 0, 1)
                        * min(lgd_base * lgd_m, 1.0)
                    ).sum()
                )
                row.append(round(stressed_ecl, 0))
            matrix.append(row)

        return pd.DataFrame(
            matrix,
            index=[f"PD x{m:.1f}" for m in pd_multipliers],
            columns=[f"LGD x{m:.2f}" for m in lgd_multipliers],
        )
