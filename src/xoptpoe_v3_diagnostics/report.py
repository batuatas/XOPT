"""Professor-facing report builder for the v3 SAA diagnostics package."""

from __future__ import annotations

import pandas as pd


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return 'nan'
    return f'{float(value):.{digits}f}'


def build_report(
    *,
    prediction_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    sleeve_attribution_summary: pd.DataFrame,
    best_worst_months: pd.DataFrame,
    selected_experiments: dict[str, str],
) -> str:
    pred_test = prediction_summary.loc[prediction_summary['split'].eq('test')].copy()
    port_test = portfolio_summary.loc[portfolio_summary['split'].eq('test')].copy()

    best60 = pred_test.loc[pred_test['strategy_label'].eq('best_60_predictor') & pred_test['horizon_months'].eq(60)].iloc[0]
    best120 = pred_test.loc[pred_test['strategy_label'].eq('best_120_predictor') & pred_test['horizon_months'].eq(120)].iloc[0]
    bestshared = pred_test.loc[pred_test['strategy_label'].eq('best_shared_predictor')].sort_values('horizon_months').iloc[0]
    pto = pred_test.loc[pred_test['strategy_label'].eq('pto_nn_signal')].groupby('strategy_label', as_index=False).agg(
        rmse=('rmse', 'mean'),
        corr=('corr', 'mean'),
        rank_ic_spearman_mean=('rank_ic_spearman_mean', 'mean'),
        predicted_dispersion_mean=('predicted_dispersion_mean', 'mean'),
    ).iloc[0]
    e2e = pred_test.loc[pred_test['strategy_label'].eq('e2e_nn_signal')].groupby('strategy_label', as_index=False).agg(
        rmse=('rmse', 'mean'),
        corr=('corr', 'mean'),
        rank_ic_spearman_mean=('rank_ic_spearman_mean', 'mean'),
        predicted_dispersion_mean=('predicted_dispersion_mean', 'mean'),
    ).iloc[0]

    port_rows = {row['strategy_label']: row for row in port_test.to_dict('records')}

    hardest = (
        sleeve_attribution_summary.loc[
            sleeve_attribution_summary['split'].eq('test')
            & sleeve_attribution_summary['strategy_label'].isin(['best_60_predictor', 'best_120_predictor'])
        ]
        .sort_values(['strategy_label', 'abs_total_active_contribution'], ascending=[True, False])
    )
    top_active = (
        sleeve_attribution_summary.loc[
            sleeve_attribution_summary['split'].eq('test')
            & sleeve_attribution_summary['strategy_label'].eq('combined_60_120_predictor')
        ]
        .sort_values('abs_total_active_contribution', ascending=False)
        .head(3)
    )
    month_rows = best_worst_months.loc[best_worst_months['split'].eq('test')].copy()

    lines: list[str] = []
    lines.append('# XOPTPOE v3 SAA Diagnostics Report')
    lines.append('')
    lines.append('## Scope And Caveat')
    lines.append('- These outputs are long-horizon SAA decision-period diagnostics built on the active v3 China-sleeve dataset and the current horse-race / PTO / E2E outputs.')
    lines.append('- They are not presented as a fully tradable non-overlapping monthly wealth backtest, because the labels are overlapping long-horizon annualized outcomes.')
    lines.append('- The purpose is interpretability: what the models predict, which sleeves they favor, how concentrated the allocations are, and where the apparent gains come from.')
    lines.append('')
    lines.append('## Most Informative Diagnostics For Discussion')
    lines.append('- The prediction scatter and heatmaps show whether the strongest 60m and 120m models are actually separating sleeves in economically sensible ways.')
    lines.append('- The rank-IC-over-time plot is the cleanest way to compare signal quality with the shared neural baselines.')
    lines.append('- The weight stack/heatmap and turnover-concentration panel show whether portfolio gains come from broad signal use or a small number of concentrated bets.')
    lines.append('- The active-return and sleeve-contribution diagnostics show whether the apparent portfolio edge comes from many months/sleeves or only a narrow subset.')
    lines.append('')
    lines.append('## Prediction Readout')
    lines.append(f"- Best 60m predictor: `{selected_experiments['best_60_predictor']}`. Test rmse={_fmt(best60['rmse'], 6)}, corr={_fmt(best60['corr'])}, sign_accuracy={_fmt(best60['sign_accuracy'])}, mean rank IC={_fmt(best60['rank_ic_spearman_mean'])}. Top predicted sleeve most often: {best60['top_predicted_sleeve']} ({_fmt(best60['top_predicted_sleeve_frequency'])}).")
    lines.append(f"- Best 120m predictor: `{selected_experiments['best_120_predictor']}`. Test rmse={_fmt(best120['rmse'], 6)}, corr={_fmt(best120['corr'])}, sign_accuracy={_fmt(best120['sign_accuracy'])}, mean rank IC={_fmt(best120['rank_ic_spearman_mean'])}. Top predicted sleeve most often: {best120['top_predicted_sleeve']} ({_fmt(best120['top_predicted_sleeve_frequency'])}).")
    lines.append(f"- Best shared benchmark: `{selected_experiments['best_shared_predictor']}`. Test rmse={_fmt(bestshared['rmse'], 6)}, corr={_fmt(bestshared['corr'])}, mean rank IC={_fmt(bestshared['rank_ic_spearman_mean'])}.")
    lines.append(f"- PTO_NN shared signal: avg test rmse={_fmt(pto['rmse'], 6)}, avg corr={_fmt(pto['corr'])}, avg rank IC={_fmt(pto['rank_ic_spearman_mean'])}, prediction dispersion={_fmt(pto['predicted_dispersion_mean'])}.")
    lines.append(f"- E2E_NN shared signal: avg test rmse={_fmt(e2e['rmse'], 6)}, avg corr={_fmt(e2e['corr'])}, avg rank IC={_fmt(e2e['rank_ic_spearman_mean'])}, prediction dispersion={_fmt(e2e['predicted_dispersion_mean'])}.")
    lines.append('')
    lines.append('## Portfolio Behavior Readout')
    for key in ['equal_weight', 'best_60_predictor', 'best_120_predictor', 'combined_60_120_predictor', 'best_shared_predictor', 'pto_nn_signal', 'e2e_nn_signal']:
        row = port_rows[key]
        lines.append(
            f"- {key}: avg_return={_fmt(row['avg_return'])}, vol={_fmt(row['volatility'])}, sharpe={_fmt(row['sharpe'])}, avg_turnover={_fmt(row['avg_turnover'])}, avg_max_weight={_fmt(row['avg_max_weight'])}, avg_effective_n={_fmt(row['avg_effective_n_sleeves'])}."
        )
    lines.append('')
    lines.append('## Signal Versus Risk-Control Interpretation')
    lines.append('- The strongest separate-horizon predictor remains the 60m elastic-net model. It has the best test rmse/corr mix and its portfolio diagnostics show strong active return with moderate turnover.')
    lines.append('- The current 120m winner by validation is not the cleanest out-of-sample predictor, so its portfolio should be interpreted more cautiously than the 60m winner.')
    lines.append('- E2E still looks much more like risk control than superior prediction. Its test prediction metrics remain weak, its prediction dispersion is low, and its portfolio volatility is suppressed materially more than equal weight.')
    lines.append('- The combined 60m/120m predictor helps on portfolio behavior relative to best shared, PTO_NN, and equal weight in this decision-period diagnostic, which is consistent with horizon diversification helping the signal layer.')
    lines.append('')
    lines.append('## Sleeve And Month Drivers')
    lines.append('- Largest absolute active-contribution sleeves for the combined 60m/120m signal on the test split:')
    for row in top_active.itertuples(index=False):
        lines.append(f"  - {row.sleeve_id}: total_active_contribution_vs_equal_weight={_fmt(row.total_active_contribution_vs_equal_weight)}; avg_weight={_fmt(row.avg_weight)}; top_weight_frequency={_fmt(row.top_weight_frequency)}.")
    lines.append('- Best and worst test months by strategy are included in the diagnostic tables; these are decision-period label outcomes, not tradable single-month realized PnL statements.')
    lines.append('')
    lines.append('## Concentration Readout')
    for key in ['best_60_predictor', 'combined_60_120_predictor', 'best_120_predictor', 'e2e_nn_signal']:
        row = port_rows[key]
        lines.append(
            f"- {key}: top5_positive_active_month_share={_fmt(row['top5_positive_active_month_share'])}, top2_sleeve_active_share_abs={_fmt(row['top2_sleeve_active_share_abs'])}, most frequent top-weight sleeve={row['top_weight_sleeve']} ({_fmt(row['top_weight_sleeve_frequency'])})."
        )
    lines.append('')
    lines.append('## Bottom Line')
    lines.append('- For professor/Akif discussion, the clearest story is: prediction quality is strongest in the separate 60m elastic-net setup; the combined 60m/120m signal gives the most compelling portfolio behavior among the common-allocator diagnostics; and E2E still looks primarily volatility-suppressing rather than signal-superior.')
    lines.append('- The system is not yet ready to jump directly into a POE/scenario-generation stage. The predictive layer is improving, but the 120m model choice and the translation from prediction quality into robust portfolio behavior still need more discipline before moving downstream.')
    return '\n'.join(lines) + '\n'
