"""
reclassify_and_figures_v2.py
XOPTPOE v4 — 19-dimension regime reclassification + 8 conference-quality figures
"""

import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
REPORTS = Path('/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports')
DATA    = Path('/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs')
FIGURES = REPORTS / 'figures_v2'
FIGURES.mkdir(exist_ok=True)

# ── Design system ──────────────────────────────────────────────────────────────
C_REGIME = {
    'higher_for_longer':            '#1F4E79',
    'higher_for_longer_divergent':  '#2E75B6',
    'risk_off_stress':              '#C00000',
    'high_stress_defensive':        '#7B0000',
    'bloc_divergence':              '#C8780A',
    'reflation_risk_on':            '#375623',
    'soft_landing':                 '#70AD47',
    'disinflationary_slowdown':     '#808080',
    'mixed_mid_cycle':              '#BFBFBF',
}

REGIME_LABELS = {
    'higher_for_longer':            'Higher for Longer',
    'higher_for_longer_divergent':  'Higher for Longer (Divergent)',
    'risk_off_stress':              'Risk-Off / Stress',
    'high_stress_defensive':        'High Stress Defensive',
    'bloc_divergence':              'US–EA Bloc Divergence',
    'reflation_risk_on':            'Reflation / Risk-On',
    'soft_landing':                 'Soft Landing',
    'disinflationary_slowdown':     'Disinflationary Slowdown',
    'mixed_mid_cycle':              'Mixed / Mid-Cycle',
}

ANCHOR_LABELS = {
    '2021-12-31': 'Dec 2021  (ZIRP)',
    '2022-12-31': 'Dec 2022  (Rate shock)',
    '2023-12-31': 'Dec 2023  (Higher for longer)',
    '2024-12-31': 'Dec 2024  (Normalisation)',
}

RF_RATES = {
    '2021-12-31': 0.0005,
    '2022-12-31': 0.0415,
    '2023-12-31': 0.0527,
    '2024-12-31': 0.0442,
}

DPI   = 300
INK   = '#1A1A1A'
GREY  = '#888888'
LGREY = '#CCCCCC'
BG    = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.facecolor': BG, 'figure.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': LGREY,
    'xtick.color': GREY, 'ytick.color': GREY,
    'axes.labelcolor': GREY, 'text.color': INK,
})

# ── Thresholds ─────────────────────────────────────────────────────────────────
TH = {
    'unemp_US_p25': 5.0,   'unemp_US_p75': 7.5,
    'infl_US_p25':  1.2,   'infl_US_p75':  2.8,
    'short_rate_US_p25': 0.25, 'short_rate_US_p75': 3.5,
    'us_real10y_p25': -0.3,    'us_real10y_p75': 1.5,
    'term_slope_US_p25': 0.5,  'term_slope_US_p75': 2.5,
    'unemp_EA_p25': 8.5,   'unemp_EA_p75': 11.0,
    'infl_EA_p25':  0.5,   'infl_EA_p75':  2.0,
    'short_rate_EA_p25': 0.0,  'short_rate_EA_p75': 2.5,
    'term_slope_EA_p25': 0.3,  'term_slope_EA_p75': 2.0,
    'term_slope_JP_p25': 0.1,  'term_slope_JP_p75': 1.0,
    'ig_oas_p75': 1.5,  'ig_oas_p90': 2.5,
    'vix_p75':   22.0,  'vix_p90':   30.0,
    'oil_wti_p25': 35.0,    'oil_wti_p75': 85.0,
    'usd_broad_p25': 112.0, 'usd_broad_p75': 125.0,
}

MACRO_COLS = [
    'infl_US','infl_EA','infl_JP','short_rate_US','short_rate_EA','short_rate_JP',
    'long_rate_US','long_rate_EA','long_rate_JP','term_slope_US','term_slope_EA',
    'term_slope_JP','unemp_US','unemp_EA','ig_oas','us_real10y','vix','oil_wti','usd_broad'
]

# ── Classifier ─────────────────────────────────────────────────────────────────
def classify_row(row, TH):
    m = {c: float(row[c]) for c in MACRO_COLS}

    # US bloc
    us_growth = ('high' if m['unemp_US'] <= TH['unemp_US_p25'] else
                 'low'  if m['unemp_US'] >= TH['unemp_US_p75'] else 'neutral')
    us_infl   = ('low'  if m['infl_US']  <= TH['infl_US_p25']  else
                 'high' if m['infl_US']  >= TH['infl_US_p75']  else 'neutral')
    us_tight  = (m['short_rate_US'] >= TH['short_rate_US_p75'] or
                 m['us_real10y']    >= TH['us_real10y_p75'])
    us_easy   = (m['short_rate_US'] <= TH['short_rate_US_p25'] and
                 m['us_real10y']    <= TH['us_real10y_p25'])
    us_policy = 'tight' if us_tight else 'easy' if us_easy else 'neutral'
    us_curve  = ('inverted' if m['term_slope_US'] < 0 else
                 'flat'     if m['term_slope_US'] <= TH['term_slope_US_p25'] else
                 'steep'    if m['term_slope_US'] >= TH['term_slope_US_p75'] else 'normal')

    # EA bloc
    ea_growth = ('high' if m['unemp_EA'] <= TH['unemp_EA_p25'] else
                 'low'  if m['unemp_EA'] >= TH['unemp_EA_p75'] else 'neutral')
    ea_infl   = ('low'  if m['infl_EA']  <= TH['infl_EA_p25']  else
                 'high' if m['infl_EA']  >= TH['infl_EA_p75']  else 'neutral')
    ea_policy = ('tight'   if m['short_rate_EA'] >= TH['short_rate_EA_p75'] else
                 'easy'    if m['short_rate_EA'] <= TH['short_rate_EA_p25'] else 'neutral')
    ea_curve  = ('inverted' if m['term_slope_EA'] < 0 else
                 'flat'     if m['term_slope_EA'] <= TH['term_slope_EA_p25'] else
                 'steep'    if m['term_slope_EA'] >= TH['term_slope_EA_p75'] else 'normal')

    # JP bloc
    jp_infl   = ('deflation' if m['infl_JP'] < 0.0 else
                 'low'       if m['infl_JP'] < 1.0 else
                 'high'      if m['infl_JP'] > 2.5 else 'neutral')
    jp_policy = ('nirp'  if m['short_rate_JP'] < 0.0  else
                 'zirp'  if m['short_rate_JP'] < 0.25 else
                 'tight' if m['short_rate_JP'] > 1.0  else 'neutral')
    jp_curve  = ('inverted' if m['term_slope_JP'] < 0 else
                 'flat'     if m['term_slope_JP'] <= TH['term_slope_JP_p25'] else
                 'steep'    if m['term_slope_JP'] >= TH['term_slope_JP_p75'] else 'normal')

    # Global stress
    ig_hi = m['ig_oas'] > TH['ig_oas_p90']; ig_mod = m['ig_oas'] > TH['ig_oas_p75']
    vx_hi = m['vix']    > TH['vix_p90'];    vx_mod = m['vix']    > TH['vix_p75']
    n_hi  = sum([ig_hi, vx_hi]);             n_mod  = sum([ig_mod, vx_mod])
    stress = ('high'     if n_hi >= 2 else
              'moderate' if n_mod >= 2 else
              'moderate' if (n_hi + n_mod) >= 1 else 'low')

    oil_r = ('crash' if m['oil_wti']   < TH['oil_wti_p25']   else
             'spike' if m['oil_wti']   > TH['oil_wti_p75']   else 'neutral')
    usd_r = ('strong' if m['usd_broad'] > TH['usd_broad_p75'] else
             'weak'   if m['usd_broad'] < TH['usd_broad_p25'] else 'neutral')

    # US–EA divergence
    rate_spread   = m['short_rate_EA'] - m['short_rate_US']
    infl_spread   = m['infl_EA']       - m['infl_US']
    bloc_rate_div = ('us_tighter' if rate_spread < -1.5 else
                     'ea_tighter' if rate_spread >  1.5 else 'aligned')
    bloc_infl_div = ('us_hotter'  if infl_spread < -2.0 else
                     'ea_hotter'  if infl_spread >  2.0 else 'aligned')

    # Composite label
    if stress == 'high' and us_growth == 'low':
        label = 'high_stress_defensive'
    elif stress == 'high':
        label = 'risk_off_stress'
    elif us_infl in ('high', 'neutral') and us_policy == 'tight':
        label = ('higher_for_longer_divergent' if bloc_rate_div == 'us_tighter'
                 else 'higher_for_longer')
    elif (bloc_rate_div != 'aligned' or bloc_infl_div != 'aligned') and stress != 'high':
        label = 'bloc_divergence'
    elif us_infl == 'low' and us_policy == 'easy' and us_growth in ('neutral', 'high'):
        label = 'soft_landing'
    elif us_infl in ('high', 'neutral') and us_policy in ('neutral', 'easy') and stress == 'low':
        label = 'reflation_risk_on'
    elif us_growth == 'low' and us_infl == 'low' and stress in ('low', 'moderate'):
        label = 'disinflationary_slowdown'
    elif stress == 'moderate' and us_growth == 'low':
        label = 'risk_off_stress'
    else:
        label = 'mixed_mid_cycle'

    fin_cond = ('loose' if stress == 'low' and us_policy == 'easy' else
                'tight' if stress in ('high', 'moderate') or us_policy == 'tight'
                else 'neutral')

    return {
        'regime_label': label,
        'dim_growth': us_growth, 'dim_inflation': us_infl,
        'dim_policy': us_policy, 'dim_stress': stress, 'dim_fin_cond': fin_cond,
        'us_growth': us_growth, 'us_inflation': us_infl,
        'us_policy': us_policy, 'us_curve': us_curve,
        'ea_growth': ea_growth, 'ea_inflation': ea_infl,
        'ea_policy': ea_policy, 'ea_curve': ea_curve,
        'jp_inflation': jp_infl, 'jp_policy': jp_policy, 'jp_curve': jp_curve,
        'global_stress': stress, 'global_oil': oil_r, 'global_usd': usd_r,
        'bloc_rate_div': bloc_rate_div, 'bloc_infl_div': bloc_infl_div,
    }

# ── Helper ─────────────────────────────────────────────────────────────────────
def savefig(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / f'{name}.{ext}', dpi=DPI, bbox_inches='tight',
                    facecolor=BG)
    plt.close(fig)

def add_takeaway(fig, text):
    fig.text(0.5, -0.02, text, ha='center', fontsize=10, color=GREY,
             style='italic', transform=fig.transFigure)

def regime_color(r):
    return C_REGIME.get(r, '#BFBFBF')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Reclassify
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
results_q123 = pd.read_csv(REPORTS / 'scenario_results_v4.csv')
results_akif = pd.read_csv(REPORTS / 'scenario_results_v4_akif.csv')
results_all  = pd.concat([results_q123, results_akif], ignore_index=True)
fm           = pd.read_parquet(DATA / 'feature_master_monthly.parquet')

print(f"  results_all: {len(results_all)} rows")
print(f"  fm: {len(fm)} rows")

# Fill missing macro columns with neutral defaults
if 'oil_wti' not in results_all.columns:
    results_all['oil_wti'] = 50.0
if 'usd_broad' not in results_all.columns:
    results_all['usd_broad'] = 118.0

print("Reclassifying all rows...")
clf_cols = [
    'regime_label','dim_growth','dim_inflation','dim_policy','dim_stress','dim_fin_cond',
    'us_growth','us_inflation','us_policy','us_curve',
    'ea_growth','ea_inflation','ea_policy','ea_curve',
    'jp_inflation','jp_policy','jp_curve',
    'global_stress','global_oil','global_usd','bloc_rate_div','bloc_infl_div',
]
# Drop old classification columns before re-adding
drop_cols = [c for c in clf_cols if c in results_all.columns]
results_all = results_all.drop(columns=drop_cols)

clf_results = results_all.apply(lambda row: classify_row(row, TH), axis=1, result_type='expand')
results_all = pd.concat([results_all, clf_results], axis=1)

# ── Anchor regime classification ───────────────────────────────────────────────
print("Classifying anchor m0 rows...")
fm_dedup = fm.drop_duplicates(subset=['month_end']).copy()
fm_dedup['month_end_str'] = fm_dedup['month_end'].dt.strftime('%Y-%m-%d')

anchor_dates = ['2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31']
anchor_regime_map = {}

for ad in anchor_dates:
    sub = fm_dedup[fm_dedup['month_end_str'] == ad]
    if len(sub) == 0:
        print(f"  WARNING: no fm row for {ad}, using neutral")
        anchor_regime_map[ad] = 'mixed_mid_cycle'
        continue
    row = sub.iloc[0].copy()
    # Fill missing macro cols with neutral
    if 'oil_wti' not in row.index or pd.isna(row.get('oil_wti', None)):
        row['oil_wti'] = 50.0
    if 'usd_broad' not in row.index or pd.isna(row.get('usd_broad', None)):
        row['usd_broad'] = 118.0
    clf = classify_row(row, TH)
    anchor_regime_map[ad] = clf['regime_label']
    print(f"  {ad}: anchor_regime = {clf['regime_label']}")

# Broadcast anchor_regime
results_all['anchor_date_str'] = results_all['anchor_date'].astype(str)
results_all['anchor_regime'] = results_all['anchor_date_str'].map(anchor_regime_map)

# Recompute regime_transition
results_all['regime_transition'] = results_all.apply(
    lambda r: 'same_regime' if r['anchor_regime'] == r['regime_label']
              else f"{r['anchor_regime']} -> {r['regime_label']}", axis=1
)

# Drop helper column
results_all = results_all.drop(columns=['anchor_date_str'])

# Save
out_path = REPORTS / 'scenario_results_all_reclassified.csv'
results_all.to_csv(out_path, index=False)
print(f"Saved {len(results_all)} rows to {out_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Figures
# ══════════════════════════════════════════════════════════════════════════════

df = results_all.copy()
df['anchor_date'] = df['anchor_date'].astype(str)

ANCHOR_ORDER = ['2021-12-31', '2022-12-31', '2023-12-31', '2024-12-31']
REGIME_ORDER = list(C_REGIME.keys())

# Weight columns
EQ_COLS  = ['w_EQ_US','w_EQ_EZ','w_EQ_JP','w_EQ_CN','w_EQ_EM']
FI_COLS  = ['w_FI_UST','w_FI_EU_GOVT']
CR_COLS  = ['w_CR_US_IG','w_CR_EU_IG','w_CR_US_HY']
RE_COLS  = ['w_RE_US','w_LISTED_RE','w_LISTED_INFRA']
ALT_COLS = ['w_ALT_GLD']
ALL_W    = EQ_COLS + FI_COLS + CR_COLS + RE_COLS + ALT_COLS

SLEEVE_COLORS = {
    'w_EQ_US':       '#1F4E79',
    'w_EQ_EZ':       '#2E75B6',
    'w_EQ_JP':       '#9DC3E6',
    'w_EQ_CN':       '#BDD7EE',
    'w_EQ_EM':       '#DEEAF1',
    'w_FI_UST':      '#375623',
    'w_FI_EU_GOVT':  '#70AD47',
    'w_CR_US_IG':    '#C8780A',
    'w_CR_EU_IG':    '#F4B183',
    'w_CR_US_HY':    '#ED7D31',
    'w_RE_US':       '#7B3F00',
    'w_LISTED_RE':   '#A0522D',
    'w_LISTED_INFRA':'#D2B48C',
    'w_ALT_GLD':     '#FFD700',
}

SLEEVE_SHORT = {
    'w_EQ_US':'EQ US','w_EQ_EZ':'EQ EZ','w_EQ_JP':'EQ JP',
    'w_EQ_CN':'EQ CN','w_EQ_EM':'EQ EM',
    'w_FI_UST':'FI UST','w_FI_EU_GOVT':'FI EU Govt',
    'w_CR_US_IG':'CR US IG','w_CR_EU_IG':'CR EU IG','w_CR_US_HY':'CR US HY',
    'w_RE_US':'RE US','w_LISTED_RE':'Listed RE','w_LISTED_INFRA':'Listed Infra',
    'w_ALT_GLD':'Gold',
}

# ── Figure 1 ───────────────────────────────────────────────────────────────────
print("Building Figure 1...")
q3_ids = ['Q1_gold_favorable', 'Q2_ew_deviation', 'Q3_house_view_7pct_total']
panel_titles = {
    'Q1_gold_favorable':         'Q1: Gold Favorable',
    'Q2_ew_deviation':           'Q2: EW Deviation',
    'Q3_house_view_7pct_total':  'Q3: House View 7% Total',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True)
fig.subplots_adjust(wspace=0.05)

# Which regimes appear in data
present_regimes = [r for r in REGIME_ORDER if r in df['regime_label'].unique()]

for ax, qid in zip(axes, q3_ids):
    sub = df[df['question_id'] == qid]
    for i, ad in enumerate(ANCHOR_ORDER):
        s2 = sub[sub['anchor_date'] == ad]
        n  = len(s2)
        bottoms = np.zeros(1)
        for regime in present_regimes:
            cnt  = (s2['regime_label'] == regime).sum()
            frac = cnt / n if n > 0 else 0
            ax.bar(i, frac, bottom=bottoms[0],
                   color=C_REGIME[regime], width=0.6, zorder=2)
            bottoms[0] += frac
        ax.text(i, 1.01, f'n={n}', ha='center', va='bottom',
                fontsize=8, color=GREY)

    ax.set_xticks(range(4))
    ax.set_xticklabels([ANCHOR_LABELS[a] for a in ANCHOR_ORDER],
                       rotation=30, ha='right', fontsize=8)
    ax.set_title(panel_titles[qid], fontsize=11, color=INK, pad=6)
    ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)
    ax.set_ylim(0, 1.12)
    if ax == axes[0]:
        ax.set_ylabel('Share of 600 samples', color=GREY)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

patches = [mpatches.Patch(color=C_REGIME[r], label=REGIME_LABELS[r])
           for r in present_regimes]
fig.legend(handles=patches, loc='lower center', ncol=5, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, -0.12))

add_takeaway(fig, ("The same portfolio model produces fundamentally different macro regimes depending on what you ask — "
                   "gold activation lives in reflation and divergence; the 7% return target lives in higher-for-longer."))
savefig(fig, 'fig01_regime_distribution')

# ── Figure 2 ───────────────────────────────────────────────────────────────────
print("Building Figure 2...")
ANCHOR_EXCESS = {
    '2021-12-31': 0.0201,
    '2022-12-31': 0.0330,
    '2023-12-31': 0.0282,
    '2024-12-31': 0.0224,
}

sub_q123 = df[df['question_id'].isin(q3_ids)].copy()

fig, axes = plt.subplots(1, 4, figsize=(16, 7), sharey=False)
fig.subplots_adjust(wspace=0.25)

for ax, ad in zip(axes, ANCHOR_ORDER):
    s = sub_q123[sub_q123['anchor_date'] == ad].copy()
    rf = RF_RATES[ad]
    colors_pts = s['regime_label'].map(lambda r: C_REGIME.get(r, '#BFBFBF'))
    ax.scatter(s['pred_return_excess']*100, s['pred_return_total']*100,
               c=colors_pts, alpha=0.3, s=6, rasterized=True, zorder=2)

    # Diagonal reference
    x_min = s['pred_return_excess'].min()*100
    x_max = s['pred_return_excess'].max()*100
    xs = np.linspace(x_min, x_max, 100)
    ys = xs + rf*100
    ax.plot(xs, ys, 'k--', lw=1, label=f'rf = {rf*100:.1f}%', zorder=3)
    ax.legend(frameon=False, fontsize=7, loc='upper left')

    # m0 point
    target_excess = ANCHOR_EXCESS[ad]
    idx = (s['pred_return_excess'] - target_excess).abs().idxmin()
    m0  = s.loc[idx]
    ax.scatter(m0['pred_return_excess']*100, m0['pred_return_total']*100,
               marker='D', s=100, color='k', zorder=5, label='m0')
    ax.annotate('m0', xy=(m0['pred_return_excess']*100, m0['pred_return_total']*100),
                xytext=(5, 5), textcoords='offset points', fontsize=8, color=INK)

    ax.set_title(ANCHOR_LABELS[ad], fontsize=9, color=INK)
    ax.set_xlabel('Excess Return (%)', color=GREY, fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('Total Return (%)', color=GREY, fontsize=8)
    ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

add_takeaway(fig, ("The model's excess return ceiling is ~3.8% — but total return reaches 10% at the 2022 anchor "
                   "because the risk-free rate adds 4.15%."))
savefig(fig, 'fig02_rf_addback')

# ── Figure 3 ───────────────────────────────────────────────────────────────────
print("Building Figure 3...")
q1 = df[df['question_id'] == 'Q1_gold_favorable'].copy()

GOLD_M0 = {'2021-12-31': 8.1, '2022-12-31': 22.3, '2023-12-31': 22.4, '2024-12-31': 23.2}

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(wspace=0.3)

# Left: scatter real10y vs gold
colors_q1 = q1['regime_label'].map(lambda r: C_REGIME.get(r, '#BFBFBF'))
ax_l.scatter(q1['us_real10y'], q1['w_ALT_GLD']*100,
             c=colors_q1, alpha=0.3, s=8, rasterized=True, zorder=2)

# OLS
valid = q1[['us_real10y','w_ALT_GLD']].dropna()
coeffs = np.polyfit(valid['us_real10y'], valid['w_ALT_GLD']*100, 1)
xr = np.linspace(valid['us_real10y'].min(), valid['us_real10y'].max(), 100)
ax_l.plot(xr, np.polyval(coeffs, xr), color=INK, lw=1.5, zorder=4, label='OLS fit')

ax_l.axvline(0, color='#C00000', lw=1, ls='--', label='Real rate = 0', zorder=3)
ax_l.annotate('Real rate < 0\n→ gold activates',
              xy=(-0.05, q1['w_ALT_GLD'].quantile(0.75)*100),
              xytext=(-2.5, q1['w_ALT_GLD'].quantile(0.6)*100),
              fontsize=8, color='#C00000', arrowprops=dict(arrowstyle='->', color='#C00000'))
ax_l.set_xlabel('US Real 10Y (%)', color=GREY)
ax_l.set_ylabel('Gold allocation (%)', color=GREY)
ax_l.set_title('Real Rates vs Gold Allocation', color=INK)
ax_l.legend(frameon=False, fontsize=8)
ax_l.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

# Right: horizontal box plots by regime, sorted by median
regime_golds = {}
for r in present_regimes:
    vals = q1[q1['regime_label'] == r]['w_ALT_GLD']*100
    if len(vals) >= 5:
        regime_golds[r] = vals.values

sorted_regimes_g = sorted(regime_golds.keys(),
                           key=lambda r: np.median(regime_golds[r]), reverse=True)

bplot = ax_r.boxplot([regime_golds[r] for r in sorted_regimes_g],
                     vert=False, patch_artist=True, notch=False,
                     medianprops=dict(color='white', lw=2))
for patch, r in zip(bplot['boxes'], sorted_regimes_g):
    patch.set_facecolor(C_REGIME.get(r, '#BFBFBF'))
    patch.set_alpha(0.8)

ax_r.set_yticks(range(1, len(sorted_regimes_g)+1))
ax_r.set_yticklabels([REGIME_LABELS.get(r, r) for r in sorted_regimes_g], fontsize=8)

# Reference lines for anchor gold weights
ref_colors = {'2021': '#888888','2022': '#444444','2023': '#222222','2024': '#000000'}
for i, (yr, w) in enumerate(sorted(GOLD_M0.items(), key=lambda x: x[0])):
    yr_short = yr[:4]
    ax_r.axvline(w, color=list(ref_colors.values())[i], ls=':', lw=1, alpha=0.8,
                 label=f'{yr_short}: {w:.1f}%')

ax_r.set_xlabel('Gold allocation (%)', color=GREY)
ax_r.set_title('Gold Allocation by Regime (Q1)', color=INK)
ax_r.legend(frameon=False, fontsize=7, loc='lower right')
ax_r.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

add_takeaway(fig, ("Gold is activated by negative real rates and reflationary conditions — not by stress alone. "
                   "Allocation ranges from 8% to 27% depending on the macro regime."))
savefig(fig, 'fig03_gold_macro')

# ── Figure 4 ───────────────────────────────────────────────────────────────────
print("Building Figure 4...")
mean_w = df.groupby('regime_label')[ALL_W].mean()
mean_w['total_eq'] = mean_w[EQ_COLS].sum(axis=1)
mean_w = mean_w.sort_values('total_eq', ascending=True)  # ascending for horiz bar

regimes_w = mean_w.index.tolist()
y_pos = np.arange(len(regimes_w))

fig, ax = plt.subplots(figsize=(16, 9))
lefts = np.zeros(len(regimes_w))

for col in ALL_W:
    vals = mean_w[col].values
    ec = '#C8780A' if col == 'w_ALT_GLD' else 'none'
    ax.barh(y_pos, vals, left=lefts, color=SLEEVE_COLORS[col],
            edgecolor=ec, linewidth=0.8 if col == 'w_ALT_GLD' else 0,
            label=SLEEVE_SHORT[col], height=0.7, zorder=2)
    lefts += vals

ax.axvline(1/14, color=INK, ls='--', lw=1, label='Equal weight (7.1%)', zorder=3)
ax.text(1/14 + 0.002, len(regimes_w)-0.5, 'Equal weight\n(7.1%)',
        fontsize=7, color=INK, va='top')

ax.set_yticks(y_pos)
ax.set_yticklabels([REGIME_LABELS.get(r, r) for r in regimes_w], fontsize=9)
ax.set_xlabel('Mean portfolio weight', color=GREY)
ax.set_xlim(0, 1.05)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.set_title('Mean Portfolio Weight Composition by Macro Regime', color=INK, fontsize=13)
ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

handles, labels = ax.get_legend_handles_labels()
# remove equal weight line from legend body; keep sleeves
sleeve_handles = [(h, l) for h, l in zip(handles, labels) if l in SLEEVE_SHORT.values()]
fig.legend([sh[0] for sh in sleeve_handles], [sh[1] for sh in sleeve_handles],
           loc='lower center', ncol=7, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.08))

add_takeaway(fig, ("Each macro regime produces a structurally different portfolio — higher-for-longer concentrates "
                   "in credit and fixed income; reflation and divergence spread weight across equities and gold."))
savefig(fig, 'fig04_weight_composition')

# ── Figure 5 ───────────────────────────────────────────────────────────────────
print("Building Figure 5...")
fig, axes = plt.subplots(1, 4, figsize=(16, 8), sharey=False)
fig.subplots_adjust(wspace=0.3)

for ax, ad in zip(axes, ANCHOR_ORDER):
    s = df[df['anchor_date'] == ad].copy()
    rf = RF_RATES[ad]

    # Sort regimes by median total return desc
    regime_data = {}
    for r in present_regimes:
        vals = s[s['regime_label'] == r]['pred_return_total']*100
        if len(vals) >= 5:
            regime_data[r] = vals.values

    sorted_r = sorted(regime_data.keys(),
                      key=lambda r: np.median(regime_data[r]), reverse=True)

    for i, r in enumerate(sorted_r):
        vals = regime_data[r]
        parts = ax.violinplot([vals], positions=[i], widths=0.7, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(C_REGIME.get(r, '#BFBFBF'))
            pc.set_alpha(0.7)
        for key in ['cbars','cmins','cmaxes','cmedians']:
            if key in parts:
                parts[key].set_color(INK)
                parts[key].set_linewidth(0.8)
        med = np.median(vals)
        ax.text(i, med, f'{med:.1f}%', ha='center', va='bottom',
                fontsize=6, color=INK, zorder=5)

    ax.axhline(rf*100, color='#C00000', ls='--', lw=1, label=f'rf = {rf*100:.1f}%')
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.set_xticks(range(len(sorted_r)))
    ax.set_xticklabels([REGIME_LABELS.get(r, r) for r in sorted_r],
                       rotation=45, ha='right', fontsize=6)
    ax.set_title(ANCHOR_LABELS[ad], fontsize=9, color=INK)
    if ax == axes[0]:
        ax.set_ylabel('Total Return (%)', color=GREY)
    ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

add_takeaway(fig, ("Total return varies by 8–10 percentage points across macro regimes at the same anchor — "
                   "the regime matters more than the question."))
savefig(fig, 'fig05_return_by_regime')

# ── Figure 6 ───────────────────────────────────────────────────────────────────
print("Building Figure 6...")
# Anchor m0 from fm
fm_dedup2 = fm.drop_duplicates(subset=['month_end']).copy()
fm_dedup2['month_end_str'] = fm_dedup2['month_end'].dt.strftime('%Y-%m-%d')
anchor_m0s = {}
for ad in ANCHOR_ORDER:
    sub = fm_dedup2[fm_dedup2['month_end_str'] == ad]
    if len(sub):
        anchor_m0s[ad] = sub.iloc[0]

panels = [
    ('infl_US',      'short_rate_US',  'US Inflation (%)',   'US Short Rate (%)'),
    ('vix',          'ig_oas',          'VIX',               'IG OAS (%)'),
    ('us_real10y',   'unemp_US',        'US Real 10Y (%)',   'US Unemployment (%)'),
    ('term_slope_US','infl_EA',         'US Term Slope (%)', 'EA Inflation (%)'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.subplots_adjust(hspace=0.35, wspace=0.3)
axes_flat = axes.flatten()

colors_all = df['regime_label'].map(lambda r: C_REGIME.get(r, '#BFBFBF'))

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes_flat, panels):
    ax.scatter(df[xcol], df[ycol], c=colors_all, alpha=0.12, s=4,
               rasterized=True, zorder=2)

    # Anchor stars
    for ad in ANCHOR_ORDER:
        if ad not in anchor_m0s:
            continue
        row = anchor_m0s[ad]
        xv = row.get(xcol, df[xcol].median()) if xcol in row.index else df[xcol].median()
        yv = row.get(ycol, df[ycol].median()) if ycol in row.index else df[ycol].median()
        if pd.isna(xv): xv = df[xcol].median()
        if pd.isna(yv): yv = df[ycol].median()
        ax.scatter(xv, yv, marker='*', s=200, color=INK, zorder=6)
        ax.annotate(ad[:4], xy=(xv, yv), xytext=(4, 4),
                    textcoords='offset points', fontsize=7, color=INK, zorder=6)

    ax.set_xlabel(xlabel, color=GREY, fontsize=9)
    ax.set_ylabel(ylabel, color=GREY, fontsize=9)
    ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

# Shared legend
patches_all = [mpatches.Patch(color=C_REGIME[r], label=REGIME_LABELS[r])
               for r in present_regimes]
fig.legend(handles=patches_all, loc='lower center', ncol=5, frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, -0.08))

add_takeaway(fig, ("MALA explores macro states far from the anchor — including combinations of inflation, rates, "
                   "and stress that have not occurred simultaneously in recent history."))
savefig(fig, 'fig06_macro_space')

# ── Figure 7 ───────────────────────────────────────────────────────────────────
print("Building Figure 7...")
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Top left: Portfolio Entropy (Q4)
ax = axes[0, 0]
q4 = df[df['question_id'] == 'Q4_max_diversification'].copy()
ent_data = {}
for r in present_regimes:
    vals = q4[q4['regime_label'] == r]['portfolio_entropy']
    if len(vals) >= 5:
        ent_data[r] = vals.values
sorted_ent = sorted(ent_data.keys(), key=lambda r: np.median(ent_data[r]), reverse=True)

bplot = ax.boxplot([ent_data[r] for r in sorted_ent],
                   patch_artist=True, notch=False,
                   medianprops=dict(color='white', lw=2))
for patch, r in zip(bplot['boxes'], sorted_ent):
    patch.set_facecolor(C_REGIME.get(r, '#BFBFBF'))
    patch.set_alpha(0.8)

ax.axhline(np.log(14), color='#1F4E79', ls='--', lw=1.2,
           label=f'Equal weight max ({np.log(14):.3f})')
ax.axhline(2.0, color=GREY, ls=':', lw=1.2, label='Practical threshold (2.0)')
ax.set_xticks(range(1, len(sorted_ent)+1))
ax.set_xticklabels([REGIME_LABELS.get(r, r) for r in sorted_ent],
                   rotation=45, ha='right', fontsize=7)
ax.set_title('Portfolio Entropy by Regime (Q4)', color=INK, fontsize=10)
ax.set_ylabel('Entropy (nats)', color=GREY)
ax.legend(frameon=False, fontsize=7)
ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

# Top right: Risk–Return Space (Q7)
ax = axes[0, 1]
q7 = df[df['question_id'] == 'Q7_max_sharpe_total'].copy()
colors_q7 = q7['regime_label'].map(lambda r: C_REGIME.get(r, '#BFBFBF'))
ax.scatter(q7['portfolio_risk']*100, q7['pred_return_total']*100,
           c=colors_q7, alpha=0.3, s=6, rasterized=True, zorder=2)

x_min_q7 = q7['portfolio_risk'].min()*100
x_max_q7 = q7['portfolio_risk'].max()*100
xs7 = np.linspace(max(x_min_q7, 0.1), x_max_q7, 100)
for sharpe, ls in [(0.3,'--'),(0.6,'--'),(1.0,'--')]:
    ys7 = sharpe * xs7
    ax.plot(xs7, ys7, ls=ls, lw=1, color=GREY)
    ax.text(xs7[-1], ys7[-1], f'SR={sharpe}', fontsize=7, color=GREY, va='bottom')

ax.set_xlabel('Portfolio Risk (%)', color=GREY, fontsize=9)
ax.set_ylabel('Total Return (%)', color=GREY, fontsize=9)
ax.set_title('Risk–Return Space (Q7: Max Sharpe Total)', color=INK, fontsize=10)
ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

# Bottom left: Equity Allocation vs 60% Target (Q6)
ax = axes[1, 0]
q6 = df[df['question_id'] == 'Q6_sixty_forty'].copy()
q6['total_eq'] = q6[EQ_COLS].sum(axis=1)
means = q6.groupby('anchor_date')['total_eq'].mean()
stds  = q6.groupby('anchor_date')['total_eq'].std()
x_pos = np.arange(4)
bar_vals = [means.get(ad, 0) for ad in ANCHOR_ORDER]
bar_errs = [stds.get(ad, 0) for ad in ANCHOR_ORDER]
ax.bar(x_pos, bar_vals, color='#1F4E79', alpha=0.8, width=0.6,
       yerr=bar_errs, capsize=4, zorder=2)
ax.axhline(0.60, color='#C00000', ls='--', lw=1.2, label='60% target', zorder=3)
max_val = max(bar_vals)
max_idx = bar_vals.index(max_val)
ax.annotate(f'Max: {max_val:.1%}',
            xy=(x_pos[max_idx], max_val),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=8, color=INK)
ax.set_xticks(x_pos)
ax.set_xticklabels([ANCHOR_LABELS[a] for a in ANCHOR_ORDER], rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Total Equity Weight', color=GREY)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax.set_title('Equity Allocation vs 60% Target (Q6)', color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

# Bottom right: US Equity Weight by Regime (Q8)
ax = axes[1, 1]
q8 = df[df['question_id'] == 'Q8_max_equity_tilt'].copy()
eq_us_data = {}
for r in present_regimes:
    vals = q8[q8['regime_label'] == r]['w_EQ_US']*100
    if len(vals) >= 5:
        eq_us_data[r] = vals.values
sorted_eq8 = sorted(eq_us_data.keys(),
                    key=lambda r: np.median(eq_us_data[r]), reverse=True)

bplot2 = ax.boxplot([eq_us_data[r] for r in sorted_eq8],
                    vert=False, patch_artist=True, notch=False,
                    medianprops=dict(color='white', lw=2))
for patch, r in zip(bplot2['boxes'], sorted_eq8):
    patch.set_facecolor(C_REGIME.get(r, '#BFBFBF'))
    patch.set_alpha(0.8)

ax.set_yticks(range(1, len(sorted_eq8)+1))
ax.set_yticklabels([REGIME_LABELS.get(r, r) for r in sorted_eq8], fontsize=7)
ax.set_xlabel('US Equity Weight (%)', color=GREY)
ax.set_title('US Equity Weight by Regime (Q8)', color=INK, fontsize=10)
ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)

add_takeaway(fig, ("Structural limits: equity tops out at ~52%, excess return at ~3.8%, entropy at 2.2 nats. "
                   "Total return reaches 10% only when the risk-free rate is elevated."))
savefig(fig, 'fig07_model_limits')

# ── Figure 8 ───────────────────────────────────────────────────────────────────
print("Building Figure 8...")
q3_labels = {
    'Q1_gold_favorable':        'Q1: Gold Favorable',
    'Q2_ew_deviation':          'Q2: EW Deviation',
    'Q3_house_view_7pct_total': 'Q3: House View 7%',
}

# Determine full set of regimes (rows and cols)
all_regimes_union = sorted(df['regime_label'].unique())
all_ar_union      = sorted(df['anchor_regime'].unique())

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.subplots_adjust(wspace=0.4)

from matplotlib.colors import Normalize
from matplotlib.cm import Blues

for ax, qid in zip(axes, q3_ids):
    sub = df[df['question_id'] == qid]
    pivot = sub.groupby(['anchor_regime','regime_label']).size().unstack(fill_value=0)
    # Reindex
    pivot = pivot.reindex(index=all_ar_union, columns=all_regimes_union, fill_value=0)
    pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)

    n_rows, n_cols = pivot.shape
    data = pivot.values.astype(float)
    norm = Normalize(vmin=0, vmax=data.max() if data.max() > 0 else 1)
    cmap = Blues

    for i in range(n_rows):
        for j in range(n_cols):
            row_lbl = pivot.index[i]
            col_lbl = pivot.columns[j]
            val = int(data[i, j])
            if row_lbl == col_lbl:
                fc = '#E8E8E8'
            else:
                fc = cmap(norm(val)) if val > 0 else 'white'
            rect = mpatches.FancyBboxPatch(
                (j - 0.5, i - 0.5), 1, 1,
                boxstyle='square,pad=0', fc=fc, ec=LGREY, lw=0.5
            )
            ax.add_patch(rect)
            if val > 0:
                text_color = 'white' if norm(val) > 0.6 and row_lbl != col_lbl else INK
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=8, color=text_color)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([REGIME_LABELS.get(c, c) for c in pivot.columns],
                       rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([REGIME_LABELS.get(r, r) for r in pivot.index], fontsize=6)
    ax.set_xlabel('Scenario Regime', color=GREY, fontsize=8)
    if ax == axes[0]:
        ax.set_ylabel('Anchor Regime', color=GREY, fontsize=8)
    ax.set_title(q3_labels[qid], color=INK, fontsize=10)
    ax.invert_yaxis()

add_takeaway(fig, ("Q1 (gold) forces the largest regime transitions — gold activation requires a genuine macro "
                   "shift away from the anchor state."))
savefig(fig, 'fig08_regime_transitions')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Output confirmation
# ══════════════════════════════════════════════════════════════════════════════
vc = results_all['regime_label'].value_counts()
total = len(results_all)

print(f"""
Reclassification complete:
  scenario_results_all_reclassified.csv — {total:,} rows
  New regime distribution:""")
for regime, cnt in vc.items():
    pct = cnt / total * 100
    print(f"    {regime}: {cnt} ({pct:.1f}%)")

print(f"""
Figure inventory (workspace_v4/reports/figures_v2/):
  fig01_regime_distribution.png + .pdf  ✅
  fig02_rf_addback.png + .pdf           ✅
  fig03_gold_macro.png + .pdf           ✅
  fig04_weight_composition.png + .pdf   ✅
  fig05_return_by_regime.png + .pdf     ✅
  fig06_macro_space.png + .pdf          ✅
  fig07_model_limits.png + .pdf         ✅
  fig08_regime_transitions.png + .pdf   ✅
""")
