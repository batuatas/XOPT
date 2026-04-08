"""Long-horizon target construction for XOPTPOE v2."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd



def _forward_max_drawdown(window_returns: pd.Series) -> float:
    wealth = (1.0 + window_returns.fillna(0.0)).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min()) if not drawdown.empty else float("nan")



def build_long_horizon_targets(
    *,
    month_end_prices: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    tb3ms_monthly: pd.DataFrame,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Build forward long-horizon targets for each sleeve-month-horizon tuple."""
    prices = month_end_prices.copy()
    prices["month_end"] = pd.to_datetime(prices["month_end"])

    rets = monthly_returns.copy()
    rets["month_end"] = pd.to_datetime(rets["month_end"])

    rf = tb3ms_monthly.copy()
    rf["month_end"] = pd.to_datetime(rf["month_end"])
    rf = rf.sort_values("month_end").reset_index(drop=True)
    rf["rf_1m"] = pd.to_numeric(rf["tb3ms"], errors="coerce") / 1200.0

    price_pivot = prices.pivot(index="month_end", columns="sleeve_id", values="adj_close").sort_index()
    ret_pivot = rets.pivot(index="month_end", columns="sleeve_id", values="ret_1m_realized").sort_index()

    months = pd.Index(sorted(price_pivot.index.unique()))
    rf = rf.set_index("month_end").reindex(months)

    rows: list[dict[str, object]] = []
    for sleeve_id in price_pivot.columns:
        sleeve_prices = price_pivot[sleeve_id]
        sleeve_rets = ret_pivot[sleeve_id].reindex(months)
        for idx, month_end in enumerate(months):
            price_t = sleeve_prices.iloc[idx]
            for horizon in horizons:
                target_idx = idx + horizon
                target_available = bool(
                    pd.notna(price_t)
                    and target_idx < len(months)
                    and pd.notna(sleeve_prices.iloc[target_idx])
                )
                gross_total = np.nan
                cumulative_total = np.nan
                gross_rf = np.nan
                cumulative_rf = np.nan
                annualized_total = np.nan
                annualized_rf = np.nan
                annualized_excess = np.nan
                realized_vol = np.nan
                realized_maxdd = np.nan
                valid_months = 0

                if target_available:
                    forward_rf = rf["rf_1m"].iloc[idx + 1 : target_idx + 1]
                    forward_rets = sleeve_rets.iloc[idx + 1 : target_idx + 1]
                    if forward_rf.notna().all() and forward_rets.notna().all():
                        gross_total = float(sleeve_prices.iloc[target_idx] / price_t)
                        cumulative_total = gross_total - 1.0
                        gross_rf = float((1.0 + forward_rf).prod())
                        cumulative_rf = gross_rf - 1.0
                        annualized_total = float(gross_total ** (12.0 / horizon) - 1.0)
                        annualized_rf = float(gross_rf ** (12.0 / horizon) - 1.0)
                        annualized_excess = float((gross_total / gross_rf) ** (12.0 / horizon) - 1.0)
                        realized_vol = float(forward_rets.std(ddof=1) * sqrt(12.0))
                        realized_maxdd = _forward_max_drawdown(forward_rets)
                        valid_months = int(forward_rets.notna().sum())
                    else:
                        target_available = False

                rows.append(
                    {
                        "sleeve_id": sleeve_id,
                        "month_end": month_end,
                        "horizon_months": horizon,
                        "horizon_years": horizon / 12.0,
                        "target_available_flag": int(target_available),
                        "forward_valid_month_count": valid_months,
                        "gross_total_forward_return": gross_total,
                        "cumulative_total_forward_return": cumulative_total,
                        "gross_rf_forward_return": gross_rf,
                        "cumulative_rf_forward_return": cumulative_rf,
                        "annualized_total_forward_return": annualized_total,
                        "annualized_rf_forward_return": annualized_rf,
                        "annualized_excess_forward_return": annualized_excess,
                        "realized_forward_volatility": realized_vol,
                        "realized_forward_max_drawdown": realized_maxdd,
                    }
                )

    out = pd.DataFrame(rows).sort_values(["sleeve_id", "month_end", "horizon_months"]).reset_index(drop=True)
    if out.duplicated(subset=["sleeve_id", "month_end", "horizon_months"]).any():
        raise ValueError("target_panel_long_horizon has duplicate keys")
    return out
