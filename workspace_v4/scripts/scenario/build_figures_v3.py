#!/usr/bin/env python3
"""build_figures_v3.py — Conference figures for investment committee."""
from __future__ import annotations
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as MPath
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPORTS = Path('/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/reports')
DATA    = Path('/Users/batuhanatas/Desktop/XOPTPOE/workspace_v4/data_refs')
FIGURES = REPORTS / 'figures_v3'
FIGURES.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(REPORTS / 'scenario_results_all_reclassified.csv')
df['anchor_date'] = df['anchor_date'].astype(str)

q123 = df[df['question_id'].isin(['Q1_gold_favorable','Q2_ew_deviation','Q3_house_view_7pct_total'])]
q1   = df[df['question_id'] == 'Q1_gold_favorable']
q2   = df[df['question_id'] == 'Q2_ew_deviation']
q3   = df[df['question_id'] == 'Q3_house_view_7pct_total']
akif = df[df['question_id'].isin(['Q4_max_diversification','Q5_max_risk',
                                   'Q6_sixty_forty','Q7_max_sharpe_total','Q8_max_equity_tilt'])]

ANCHORS = ['2021-12-31','2022-12-31','2023-12-31','2024-12-31']
ANCHOR_SHORT = {'2021-12-31':'2021','2022-12-31':'2022','2023-12-31':'2023','2024-12-31':'2024'}
ANCHOR_LABEL = {
    '2021-12-31': 'Dec 2021\n(ZIRP — rf=0.05%)',
    '2022-12-31': 'Dec 2022\n(Rate shock — rf=4.15%)',
    '2023-12-31': 'Dec 2023\n(Higher for longer — rf=5.27%)',
    '2024-12-31': 'Dec 2024\n(Normalisation — rf=4.42%)',
}
RF = {'2021-12-31':0.0005,'2022-12-31':0.0415,'2023-12-31':0.0527,'2024-12-31':0.0442}

M0 = {
    '2021-12-31': {'infl_US':7.0,'short_rate_US':0.05,'us_real10y':-0.32,'vix':19.7,
                   'ig_oas':1.76,'usd_broad':117.7,'oil_wti':41.2,'unemp_US':3.9,'w_ALT_GLD':0.081},
    '2022-12-31': {'infl_US':6.5,'short_rate_US':4.15,'us_real10y':1.42,'vix':25.8,
                   'ig_oas':1.89,'usd_broad':121.3,'oil_wti':80.5,'unemp_US':3.5,'w_ALT_GLD':0.223},
    '2023-12-31': {'infl_US':3.4,'short_rate_US':5.27,'us_real10y':2.05,'vix':12.5,
                   'ig_oas':1.24,'usd_broad':119.8,'oil_wti':71.6,'unemp_US':3.7,'w_ALT_GLD':0.224},
    '2024-12-31': {'infl_US':2.9,'short_rate_US':4.42,'us_real10y':2.11,'vix':15.6,
                   'ig_oas':1.08,'usd_broad':120.4,'oil_wti':70.1,'unemp_US':4.2,'w_ALT_GLD':0.232},
}

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
C = {
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
RL = {
    'higher_for_longer':            'Higher for Longer',
    'higher_for_longer_divergent':  'Higher for Longer\n(US–EA Divergence)',
    'risk_off_stress':              'Risk-Off / Stress',
    'high_stress_defensive':        'High Stress\nDefensive',
    'bloc_divergence':              'US–EA Bloc\nDivergence',
    'reflation_risk_on':            'Reflation /\nRisk-On',
    'soft_landing':                 'Soft Landing',
    'disinflationary_slowdown':     'Disinflationary\nSlowdown',
    'mixed_mid_cycle':              'Mixed /\nMid-Cycle',
}

INK='#1A1A1A'; GREY='#888888'; LGREY='#DDDDDD'; BG='#FFFFFF'
DPI=300

ANCHOR_COLORS = {
    '2021-12-31': '#888888',
    '2022-12-31': '#C8780A',
    '2023-12-31': '#2E75B6',
    '2024-12-31': '#1F4E79',
}

plt.rcParams.update({
    'font.family':'sans-serif','font.size':10,
    'axes.facecolor':BG,'figure.facecolor':BG,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':LGREY,'axes.labelcolor':GREY,
    'xtick.color':GREY,'ytick.color':GREY,'text.color':INK,
})

def save(fig, name):
    fig.savefig(FIGURES / f'{name}.png', dpi=DPI, bbox_inches='tight', facecolor=BG)
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  {name} OK')

def legend_patches(regimes):
    return [mpatches.Patch(color=C.get(r,'#CCC'), label=RL.get(r,r)) for r in regimes if r in C]

def add_grid(ax):
    ax.yaxis.grid(True, color=LGREY, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

# ---------------------------------------------------------------------------
# Figure 1 — Question regime compass (bubble chart)
# ---------------------------------------------------------------------------
print("Building Figure 1...")
fig, axes = plt.subplots(1, 3, figsize=(16, 9), facecolor=BG)
fig.subplots_adjust(wspace=0.05)

questions = ['Q1_gold_favorable', 'Q2_ew_deviation', 'Q3_house_view_7pct_total']
q_titles  = ['Q1: Gold-Favorable Macro', 'Q2: Maximum Concentration', 'Q3: 7% Total Return']

all_regimes_used = sorted(df[df['question_id'].isin(questions)]['regime_label'].unique())
regime_y = {r: i for i, r in enumerate(reversed(all_regimes_used))}

for ax_idx, (qid, qtitle) in enumerate(zip(questions, q_titles)):
    ax = axes[ax_idx]
    sub = df[df['question_id'] == qid]

    for a_idx, anchor in enumerate(ANCHORS):
        asub = sub[sub['anchor_date'] == anchor]
        if asub.empty:
            continue
        total = len(asub)
        for regime, rgrp in asub.groupby('regime_label'):
            share = len(rgrp) / total
            y = regime_y.get(regime, 0)
            x = a_idx
            size = max(share * 2000, 20)
            color = C.get(regime, '#CCC')
            ax.scatter(x, y, s=size, color=color, alpha=0.85, zorder=3)
            if share >= 0.10:
                ax.text(x, y, f'{share*100:.0f}%', ha='center', va='center',
                        fontsize=7, color='white' if share > 0.3 else INK,
                        fontweight='bold', zorder=4)

    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.7, len(all_regimes_used) - 0.3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([ANCHOR_SHORT[a] for a in ANCHORS], fontsize=9)
    ax.set_xlabel('Anchor year', color=GREY, fontsize=9)
    ax.set_title(qtitle, color=INK, fontsize=11, fontweight='bold', pad=8)

    if ax_idx == 0:
        ax.set_yticks(list(regime_y.values()))
        ax.set_yticklabels([RL.get(r, r) for r in reversed(all_regimes_used)], fontsize=8)
    else:
        ax.set_yticks([])
        # thin vertical divider
        ax.axvline(-0.6, color=LGREY, lw=1)

    add_grid(ax)
    # Remove left spine for non-first panels
    if ax_idx > 0:
        ax.spines['left'].set_visible(False)

fig.text(0.5, -0.02,
    "Each bubble = share of 600 MALA samples. Regime classification uses all 19 macro dimensions including US-EA policy divergence.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig01_question_regime_compass')

# ---------------------------------------------------------------------------
# Figure 2 — Macro fingerprint (radar chart)
# ---------------------------------------------------------------------------
print("Building Figure 2...")

RADAR_VARS = ['infl_US','short_rate_US','us_real10y_inv','ig_oas','vix','usd_broad']
RADAR_LABELS = ['US Inflation', 'Policy Rate', 'Real Rate\n(inverted)', 'Credit Stress', 'VIX', 'USD Strength']

# Normalization ranges (historical min/max)
RADAR_MINMAX = {
    'infl_US':        (0.0, 10.0),
    'short_rate_US':  (0.0, 6.0),
    'us_real10y_inv': (-3.0, 2.0),   # will be inverted: high score = more negative real rate
    'ig_oas':         (0.5, 4.0),
    'vix':            (10.0, 50.0),
    'usd_broad':      (105.0, 130.0),
}

def normalize_radar(val, varname):
    lo, hi = RADAR_MINMAX[varname]
    return np.clip((val - lo) / (hi - lo), 0, 1)

def radar_values(sub, anchor):
    asub = sub[sub['anchor_date'] == anchor]
    if asub.empty:
        return None
    # lowest G 20%
    thresh = asub['G_value'].quantile(0.20)
    best   = asub[asub['G_value'] <= thresh]
    if best.empty:
        return None
    vals = {}
    vals['infl_US']        = best['infl_US'].median()
    vals['short_rate_US']  = best['short_rate_US'].median()
    # inverted real rate: we want high score for low/negative real rate
    vals['us_real10y_inv'] = -best['us_real10y'].median()   # invert
    vals['ig_oas']         = best['ig_oas'].median()
    vals['vix']            = best['vix'].median()
    vals['usd_broad']      = best['usd_broad'].median()
    return [normalize_radar(vals[v], v) for v in RADAR_VARS]

def m0_radar(anchor):
    m = M0[anchor]
    vals = {
        'infl_US':        m['infl_US'],
        'short_rate_US':  m['short_rate_US'],
        'us_real10y_inv': -m['us_real10y'],
        'ig_oas':         m['ig_oas'],
        'vix':            m['vix'],
        'usd_broad':      m['usd_broad'],
    }
    return [normalize_radar(vals[v], v) for v in RADAR_VARS]

N_SPOKES = len(RADAR_VARS)
angles   = np.linspace(0, 2*np.pi, N_SPOKES, endpoint=False).tolist()
angles  += angles[:1]  # close the loop

fig, axes = plt.subplots(1, 3, figsize=(16, 10), facecolor=BG,
                          subplot_kw=dict(polar=True))
fig.subplots_adjust(wspace=0.4)

for ax_idx, (qid, qtitle) in enumerate(zip(questions, q_titles)):
    ax = axes[ax_idx]
    sub = df[df['question_id'] == qid]

    for anchor in ANCHORS:
        rv = radar_values(sub, anchor)
        if rv is None:
            continue
        rv_plot = rv + rv[:1]
        color = ANCHOR_COLORS[anchor]
        ax.plot(angles, rv_plot, color=color, lw=1.8, label=ANCHOR_SHORT[anchor], alpha=0.9)
        ax.fill(angles, rv_plot, color=color, alpha=0.06)

    # m0 overlay (use 2024 as representative, dashed)
    m0r = m0_radar('2024-12-31')
    m0r_plot = m0r + m0r[:1]
    ax.plot(angles, m0r_plot, color=INK, lw=1.0, linestyle='--', alpha=0.4, label='m0 (2024)')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, fontsize=8, color=INK)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], fontsize=6, color=GREY)
    ax.set_ylim(0, 1)
    ax.set_title(qtitle, color=INK, fontsize=10, fontweight='bold', pad=15)
    ax.grid(color=LGREY, lw=0.5)
    ax.spines['polar'].set_color(LGREY)

    if ax_idx == 0:
        handles = [mpatches.Patch(color=ANCHOR_COLORS[a], label=ANCHOR_SHORT[a]) for a in ANCHORS]
        handles.append(mpatches.Patch(color=INK, label='m0 ref', alpha=0.4))
        ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.5, 1.15),
                  frameon=False, fontsize=8)

fig.text(0.5, -0.02,
    "Lines show median macro state of the 120 best-fit scenarios (lowest G value) per anchor. Dashed = anchor starting point.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig02_macro_fingerprint')

# ---------------------------------------------------------------------------
# Figure 3 — MALA convergence scatter
# ---------------------------------------------------------------------------
print("Building Figure 3...")
MARKER_SHAPES = {'2021-12-31':'o','2022-12-31':'^','2023-12-31':'s','2024-12-31':'D'}
Y_VARS = {
    'Q1_gold_favorable':           ('w_ALT_GLD', 'Gold allocation (%)', lambda v: v*100),
    'Q2_ew_deviation':             ('pred_return_total', 'Total return (%)', lambda v: v*100),
    'Q3_house_view_7pct_total':    ('pred_return_total', 'Total return (%)', lambda v: v*100),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 8), facecolor=BG)
fig.subplots_adjust(wspace=0.3)

for ax_idx, qid in enumerate(questions):
    ax = axes[ax_idx]
    sub = df[df['question_id'] == qid].copy()

    ycol, ylabel, yfunc = Y_VARS[qid]

    g_vals = sub['G_value'].values
    g_range = g_vals.max() - g_vals.min()
    if g_range < 1e-10:
        g_range = 1.0

    cmap = plt.cm.RdYlBu_r
    scatter_obj = None
    for anchor in ANCHORS:
        asub = sub[sub['anchor_date'] == anchor]
        if asub.empty:
            continue
        xv = asub['us_real10y'].values
        yv = yfunc(asub[ycol].values)
        gn = (asub['G_value'].values - g_vals.min()) / g_range
        scatter_obj = ax.scatter(xv, yv, c=gn, cmap=cmap, vmin=0, vmax=1,
                             alpha=0.35, s=10, marker=MARKER_SHAPES[anchor],
                             rasterized=True, zorder=2)

    # anchor m0 points
    for anchor in ANCHORS:
        m0x = M0[anchor]['us_real10y']
        if ycol == 'w_ALT_GLD':
            m0y = M0[anchor]['w_ALT_GLD'] * 100
        else:
            m0y = (RF[anchor] + 0.025) * 100  # approx m0 total return
        ax.scatter([m0x], [m0y], color=INK, s=120, marker='*', zorder=5,
                   label='m0' if anchor == '2021-12-31' else '')
        ax.annotate(ANCHOR_SHORT[anchor], (m0x, m0y),
                    xytext=(4, 4), textcoords='offset points', fontsize=7, color=INK)

    if scatter_obj is not None:
        cbar = fig.colorbar(scatter_obj, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('G value (lower = better)', fontsize=7, color=GREY)
        cbar.ax.tick_params(labelsize=6)

    ax.set_xlabel('US Real Rate (%)', color=GREY)
    ax.set_ylabel(ylabel, color=GREY)
    ax.set_title(q_titles[ax_idx], color=INK, fontsize=10, fontweight='bold')
    ax.axvline(0, color=LGREY, lw=1, linestyle='--', zorder=1)
    add_grid(ax)

    # Legend for marker shapes
    leg_handles = [plt.scatter([], [], marker=MARKER_SHAPES[a], color=GREY, s=30,
                               label=ANCHOR_SHORT[a]) for a in ANCHORS]
    ax.legend(handles=leg_handles, fontsize=7, frameon=False, loc='upper right')

fig.text(0.5, -0.02,
    "G value measures how well each scenario satisfies the probe question. MALA concentrates samples in low-G regions while respecting macro plausibility.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig03_mala_convergence')

# ---------------------------------------------------------------------------
# Figure 4a/b/c — Portfolio by scenario
# ---------------------------------------------------------------------------
print("Building Figures 4a/b/c...")

SLEEVES = ['EQ_US','EQ_EZ','EQ_JP','EQ_CN','EQ_EM',
           'FI_UST','FI_EU_GOVT',
           'CR_US_IG','CR_EU_IG','CR_US_HY',
           'RE_US','LISTED_RE','LISTED_INFRA',
           'ALT_GLD']

SLEEVE_COLORS = {
    'EQ_US':'#1F4E79','EQ_EZ':'#2E75B6','EQ_JP':'#9DC3E6','EQ_CN':'#BDD7EE','EQ_EM':'#DEEAF1',
    'FI_UST':'#375623','FI_EU_GOVT':'#70AD47',
    'CR_US_IG':'#C8780A','CR_EU_IG':'#F4B183','CR_US_HY':'#ED7D31',
    'RE_US':'#7B3F00','LISTED_RE':'#A0522D','LISTED_INFRA':'#D2B48C',
    'ALT_GLD':'#FFD700',
}

FIG4_SPECS = [
    ('Q1_gold_favorable',        'fig04a_portfolio_q1_gold',    'Portfolio under Gold-Favorable Macro',
     'Note: 2021 scenarios use the current model (trained Feb 2021). The historical benchmark chart uses a rolling model -- gold activation differs.'),
    ('Q2_ew_deviation',          'fig04b_portfolio_q2_ew',      'Portfolio under Maximum Concentration Macro', None),
    ('Q3_house_view_7pct_total', 'fig04c_portfolio_q3_return',  'Portfolio Targeting 7% Total Return', None),
]

for qid, fname, title, footnote_extra in FIG4_SPECS:
    sub = df[df['question_id'] == qid]
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=BG)

    n_anchors = 4
    n_sleeves = len(SLEEVES)
    group_width = 0.8
    bar_width = group_width / n_sleeves
    EQ_WEIGHT = 1/14

    for a_idx, anchor in enumerate(ANCHORS):
        asub = sub[sub['anchor_date'] == anchor]
        if asub.empty:
            continue
        # Lowest G 10%
        thresh = asub['G_value'].quantile(0.10)
        best   = asub[asub['G_value'] <= thresh]
        if best.empty:
            continue

        for s_idx, sleeve in enumerate(SLEEVES):
            wcol = f'w_{sleeve}'
            if wcol not in best.columns:
                continue
            mean_w = best[wcol].mean() * 100
            x = a_idx + (s_idx - n_sleeves/2 + 0.5) * bar_width
            color = SLEEVE_COLORS[sleeve]
            edge = '#C8780A' if sleeve == 'ALT_GLD' else color
            lw   = 1.5       if sleeve == 'ALT_GLD' else 0
            ax.bar(x, mean_w, width=bar_width*0.9, color=color,
                   edgecolor=edge, linewidth=lw, zorder=3)

    ax.axhline(EQ_WEIGHT * 100, color=GREY, lw=1, linestyle='--', zorder=2,
               label='Equal weight (7.1%)')
    ax.set_xticks(range(n_anchors))
    ax.set_xticklabels([ANCHOR_LABEL[a] for a in ANCHORS], fontsize=8)
    ax.set_ylabel('Mean portfolio weight (%)', color=GREY)
    ax.set_title(title, color=INK, fontsize=13, fontweight='bold')
    add_grid(ax)
    ax.legend(fontsize=8, frameon=False, loc='upper right')

    # Sleeve legend
    leg_patches = [mpatches.Patch(color=SLEEVE_COLORS[s], label=s) for s in SLEEVES]
    fig.legend(handles=leg_patches, loc='lower center', ncol=7, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.04))

    note = "Each bar group = lowest-G 10% of scenarios (best 60 per anchor). Colored by sleeve."
    if footnote_extra:
        note = footnote_extra + '\n' + note
    fig.text(0.5, -0.08 if footnote_extra else -0.02, note,
             ha='center', fontsize=8 if footnote_extra else 9, color=GREY,
             style='italic', transform=fig.transFigure, wrap=True)
    save(fig, fname)

# ---------------------------------------------------------------------------
# Figure 5 — RF story
# ---------------------------------------------------------------------------
print("Building Figure 5...")
fig = plt.figure(figsize=(16, 6), facecolor=BG)
gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.35)
ax_main = fig.add_subplot(gs[0])
ax_range = fig.add_subplot(gs[1])

BLUE_DARK = '#1F4E79'
GOLD_COL  = '#C8780A'
anchor_y  = {a: i for i, a in enumerate(ANCHORS)}

for anchor in ANCHORS:
    sub_a = q123[q123['anchor_date'] == anchor]
    if sub_a.empty:
        continue
    mean_exc = sub_a['pred_return_excess'].mean() * 100
    std_exc  = sub_a['pred_return_excess'].std()  * 100
    rf_pct   = RF[anchor] * 100
    y        = anchor_y[anchor]

    # Excess segment
    ax_main.barh(y, mean_exc, color=BLUE_DARK, height=0.4, left=0, zorder=3)
    # RF segment
    ax_main.barh(y, rf_pct, color=GOLD_COL, height=0.4, left=mean_exc, zorder=3)
    # Error bar on excess
    ax_main.errorbar(mean_exc, y, xerr=std_exc, fmt='none', color=INK, lw=1.2, capsize=4, zorder=4)
    # Annotations
    ax_main.text(mean_exc / 2, y, f'{mean_exc:.1f}%', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
    total = mean_exc + rf_pct
    ax_main.text(mean_exc + rf_pct / 2, y, f'+{rf_pct:.2f}%', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
    ax_main.text(total + 0.1, y, f'= {total:.1f}%', ha='left', va='center',
                 fontsize=9, color=INK, fontweight='bold')

ax_main.set_yticks(list(anchor_y.values()))
ax_main.set_yticklabels([ANCHOR_LABEL[a] for a in ANCHORS], fontsize=9)
ax_main.set_xlabel('Return (%/yr)', color=GREY)
ax_main.set_title('Total Return = Excess Return + Risk-Free Rate', color=INK,
                   fontsize=13, fontweight='bold')
ax_main.axvline(0, color=LGREY, lw=1)

leg = [mpatches.Patch(color=BLUE_DARK, label='Excess return (model alpha)'),
       mpatches.Patch(color=GOLD_COL, label='Risk-free rate add-back')]
ax_main.legend(handles=leg, fontsize=9, frameon=False, loc='lower right')
ax_main.xaxis.grid(True, color=LGREY, lw=0.5)
ax_main.set_axisbelow(True)

# Range strip
for anchor in ANCHORS:
    sub_a = q123[q123['anchor_date'] == anchor]
    if sub_a.empty:
        continue
    tot = sub_a['pred_return_total'] * 100
    p05, p25, p75, p95 = tot.quantile([0.05, 0.25, 0.75, 0.95])
    y = anchor_y[anchor]
    col = ANCHOR_COLORS[anchor]
    ax_range.plot([p05, p95], [y, y], color=col, lw=1, alpha=0.5)
    ax_range.plot([p25, p75], [y, y], color=col, lw=4, alpha=0.9)
    ax_range.scatter([tot.median()], [y], color=col, s=50, zorder=4)

ax_range.set_yticks(list(anchor_y.values()))
ax_range.set_yticklabels([ANCHOR_SHORT[a] for a in ANCHORS], fontsize=8)
ax_range.set_xlabel('Total return (%/yr) — 5th-95th percentile range', color=GREY, fontsize=8)
ax_range.set_title('Total Return Range (all Q1-Q3 scenarios)', color=GREY, fontsize=9)
ax_range.xaxis.grid(True, color=LGREY, lw=0.5)
ax_range.set_axisbelow(True)
ax_range.spines['left'].set_visible(False)
ax_range.spines['bottom'].set_color(LGREY)

fig.text(0.5, -0.03,
    "Excess return = model prediction above T-bills. Total return = excess + short_rate_US. Range shows 5th-95th percentile across all 1,800 scenarios per anchor.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig05_rf_story')

# ---------------------------------------------------------------------------
# Figure 6 — Macro shift (diverging bars)
# ---------------------------------------------------------------------------
print("Building Figure 6...")
MACRO_SHIFT_VARS = ['infl_US','short_rate_US','us_real10y','vix','ig_oas','usd_broad','oil_wti','unemp_US']
MACRO_UNITS = {
    'infl_US':'%','short_rate_US':'%','us_real10y':'%',
    'vix':'pts','ig_oas':'%','usd_broad':'index pts','oil_wti':'$/bbl','unemp_US':'%'
}
MACRO_LABELS_SHORT = {
    'infl_US':'US Inflation','short_rate_US':'US Policy Rate','us_real10y':'US Real Rate',
    'vix':'VIX','ig_oas':'IG OAS','usd_broad':'USD (broad)','oil_wti':'Oil (WTI)','unemp_US':'US Unemployment'
}

fig, axes = plt.subplots(1, 3, figsize=(16, 9), facecolor=BG)
fig.subplots_adjust(wspace=0.4)

for ax_idx, qid in enumerate(questions):
    ax = axes[ax_idx]
    sub = df[df['question_id'] == qid]

    for a_offset, anchor in enumerate(ANCHORS):
        asub = sub[sub['anchor_date'] == anchor]
        if asub.empty:
            continue
        thresh = asub['G_value'].quantile(0.20)
        best   = asub[asub['G_value'] <= thresh]
        if best.empty:
            continue

        m0 = M0[anchor]
        deltas = []
        for var in MACRO_SHIFT_VARS:
            if var in best.columns and var in m0:
                delta = best[var].mean() - m0[var]
                deltas.append(delta)
            else:
                deltas.append(0.0)

        y_positions = [i * 1.2 + a_offset * 0.22 for i in range(len(MACRO_SHIFT_VARS))]
        colors = ['#C00000' if d > 0 else '#1F4E79' for d in deltas]
        ax.barh(y_positions, deltas, height=0.18,
                color=[c if abs(d) > 0.01 else LGREY for c, d in zip(colors, deltas)],
                alpha=0.8, zorder=3, label=ANCHOR_SHORT[anchor] if ax_idx == 0 else '')

    # Y-axis labels at center of each variable group
    ax.set_yticks([i * 1.2 + 0.33 for i in range(len(MACRO_SHIFT_VARS))])
    ax.set_yticklabels([f"{MACRO_LABELS_SHORT[v]}\n({MACRO_UNITS[v]})" for v in MACRO_SHIFT_VARS],
                       fontsize=8)
    ax.axvline(0, color=INK, lw=0.8, zorder=4)
    ax.set_xlabel('Scenario minus Anchor m0', color=GREY)
    ax.set_title(q_titles[ax_idx], color=INK, fontsize=10, fontweight='bold')
    add_grid(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    if ax_idx == 0:
        leg_handles = [mpatches.Patch(color=ANCHOR_COLORS[a], label=ANCHOR_SHORT[a])
                       for a in ANCHORS]
        ax.legend(handles=leg_handles, fontsize=8, frameon=False, loc='lower right')

fig.text(0.5, -0.02,
    "Bars show mean macro shift from anchor m0 to the 20% of scenarios that best satisfy each question objective.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig06_macro_shift')

# ---------------------------------------------------------------------------
# Figure 7 — G function landscape
# ---------------------------------------------------------------------------
print("Building Figure 7...")
Q_ANSWERS = {
    'Q1_gold_favorable':         'Gold-favorable macro: negative real rates, low USD, moderate inflation.',
    'Q2_ew_deviation':           'Maximum concentration: tight policy, inverted curve, high credit spreads.',
    'Q3_house_view_7pct_total':  '7% total return: achievable at 2022-2024 anchors via rf; requires extreme macro at 2021.',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor=BG)
fig.subplots_adjust(wspace=0.3, bottom=0.18)

for ax_idx, qid in enumerate(questions):
    ax = axes[ax_idx]
    sub = df[df['question_id'] == qid]

    for anchor in ANCHORS:
        asub = sub[sub['anchor_date'] == anchor]
        if asub.empty:
            continue
        g_vals = asub['G_value'].values
        if len(g_vals) < 10:
            continue
        # Check for constant array
        if np.std(g_vals) < 1e-10:
            continue
        color = ANCHOR_COLORS[anchor]

        kde = gaussian_kde(g_vals, bw_method=0.3)
        xs  = np.linspace(g_vals.min(), g_vals.max(), 300)
        ys  = kde(xs)

        ax.plot(xs, ys, color=color, lw=1.8, label=ANCHOR_SHORT[anchor], zorder=3)

        # Shade below median
        med = np.median(g_vals)
        mask = xs <= med
        ax.fill_between(xs[mask], 0, ys[mask], color=color, alpha=0.12, zorder=2)

        # G(m0) vertical line: tau * TAU_DIVISOR = G(m0) since tau = G(m0)/5
        if 'tau_effective' in asub.columns:
            tau_vals = asub['tau_effective'].dropna()
            if len(tau_vals) > 0:
                g_m0 = float(tau_vals.iloc[0]) * 5.0
                ax.axvline(g_m0, color=color, lw=1, linestyle='--', alpha=0.7)
                ax.annotate('G(m0)', (g_m0, ys.max()*0.9), xytext=(3, 0),
                            textcoords='offset points', fontsize=6, color=color, rotation=90)

    ax.set_xlabel('G value', color=GREY)
    ax.set_ylabel('Density', color=GREY)
    ax.set_title(q_titles[ax_idx], color=INK, fontsize=10, fontweight='bold')
    add_grid(ax)
    ax.legend(fontsize=8, frameon=False, loc='upper right')

    # Question answer box below
    answer = Q_ANSWERS.get(qid, '')
    ax.text(0.5, -0.28, f'-> {answer}', ha='center', va='top', transform=ax.transAxes,
            fontsize=8, color=INK, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor=LGREY, lw=0.5))

    ax.text(0.98, 0.95, 'n=600 per anchor', ha='right', va='top',
            transform=ax.transAxes, fontsize=7, color=GREY)

fig.text(0.5, -0.03,
    "Lower G = scenario better satisfies the question. MALA samples proportional to exp(-G/tau). tau = G(m0)/5 (adaptive temperature).",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig07_g_landscape')

# ---------------------------------------------------------------------------
# Figure 8 — Dominant scenario answer cards (4x3 grid)
# ---------------------------------------------------------------------------
print("Building Figure 8...")
# Get best scenario per question x anchor
best_scenarios = (df.groupby(['question_id','anchor_date'])
                    .apply(lambda x: x.nsmallest(1,'G_value'))
                    .reset_index(drop=True))

Q_SHORT = {
    'Q1_gold_favorable':         'Q1: Gold-Favorable',
    'Q2_ew_deviation':           'Q2: Max Concentration',
    'Q3_house_view_7pct_total':  'Q3: 7% Return Target',
}

SLEEVE_SHORT = {s: s.replace('_','').replace('LISTED','L.') for s in SLEEVES}

fig = plt.figure(figsize=(16, 12), facecolor=BG)
fig.subplots_adjust(hspace=0.45, wspace=0.3)

for row_i, anchor in enumerate(ANCHORS):
    for col_j, qid in enumerate(questions):
        ax = fig.add_subplot(4, 3, row_i*3 + col_j + 1)

        row = best_scenarios[
            (best_scenarios['question_id'] == qid) &
            (best_scenarios['anchor_date'] == anchor)
        ]
        if row.empty:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color=GREY)
            continue

        row = row.iloc[0]
        regime = row.get('regime_label','unknown')
        regime_color = C.get(regime, '#CCC')

        # Background tint for regime
        ax.set_facecolor(regime_color + '15')

        # Regime label header
        ax.text(0.5, 0.95, RL.get(regime, regime).replace('\n',' '),
                ha='center', va='top', transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=regime_color)

        # 3 key macro numbers
        infl   = row.get('infl_US', float('nan'))
        rate   = row.get('short_rate_US', float('nan'))
        real10 = row.get('us_real10y', float('nan'))
        macro_text = f'{infl:.1f}% infl  |  {rate:.1f}% rate  |  {real10:+.2f}% real'
        ax.text(0.5, 0.80, macro_text, ha='center', va='top', transform=ax.transAxes,
                fontsize=7.5, color=INK)

        # Mini bar chart of weights
        wvals = []
        for sleeve in SLEEVES:
            wcol = f'w_{sleeve}'
            wvals.append(row.get(wcol, 0.0) * 100)

        colors_bar = [SLEEVE_COLORS[s] for s in SLEEVES]
        xs = np.arange(len(SLEEVES))
        ax2_height = 0.38
        ax2_bottom = 0.30
        # Draw bars using ax.inset_axes with relative coordinates [x0, y0, width, height]
        ax_inset = ax.inset_axes([0.02, ax2_bottom, 0.96, ax2_height])
        ax_inset.bar(xs, wvals, color=colors_bar, width=0.8, zorder=3)
        ax_inset.axhline(7.1, color=GREY, lw=0.5, linestyle='--')
        ax_inset.set_xlim(-0.5, len(SLEEVES)-0.5)
        ax_inset.set_xticks(xs)
        ax_inset.set_xticklabels([s.replace('_US','').replace('_','') for s in SLEEVES],
                                   rotation=90, fontsize=5)
        ax_inset.set_yticks([])
        ax_inset.set_facecolor('none')
        for sp in ax_inset.spines.values():
            sp.set_visible(False)

        # Total return
        tot_ret = row.get('pred_return_total', float('nan')) * 100
        ax.text(0.5, 0.27, f'{tot_ret:.1f}% total return',
                ha='center', va='top', transform=ax.transAxes,
                fontsize=10, fontweight='bold', color=INK)

        # 2021 footnote
        if anchor == '2021-12-31' and qid == 'Q1_gold_favorable':
            ax.text(0.5, 0.06, 'rolling model differs',
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=6, color='#C00000')

        ax.axis('off')

        # Column title (first row only)
        if row_i == 0:
            ax.set_title(Q_SHORT.get(qid, qid), color=INK, fontsize=9, fontweight='bold', pad=4)

        # Row label (first column only)
        if col_j == 0:
            ax.text(-0.08, 0.5, ANCHOR_SHORT[anchor], ha='right', va='center',
                    transform=ax.transAxes, fontsize=9, fontweight='bold', color=INK,
                    rotation=90)

fig.text(0.5, -0.02,
    "Each cell shows the single macro scenario that best satisfies the question objective (lowest G value). Portfolio weights are the benchmark's allocation at that macro state.",
    ha='center', fontsize=9, color=GREY, style='italic', transform=fig.transFigure)
save(fig, 'fig08_dominant_scenario')

# ---------------------------------------------------------------------------
# Figure 9 — Benchmark vs MALA (2021 gold inconsistency)
# ---------------------------------------------------------------------------
print("Building Figure 9...")
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.subplots_adjust(wspace=0.35)

# Left panel: anchor m0 gold weight (current model)
m0_gold = [M0[a]['w_ALT_GLD'] * 100 for a in ANCHORS]
bar_colors = [ANCHOR_COLORS[a] for a in ANCHORS]
bars = ax_left.bar(range(4), m0_gold, color=bar_colors, alpha=0.85, width=0.5, zorder=3)
for i, (v, anchor) in enumerate(zip(m0_gold, ANCHORS)):
    ax_left.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold', color=INK)
ax_left.set_xticks(range(4))
ax_left.set_xticklabels([ANCHOR_LABEL[a] for a in ANCHORS], fontsize=8)
ax_left.set_ylabel('ALT_GLD weight (%)', color=GREY)
ax_left.set_title('Benchmark at Anchor m0\n(Current Model, Trained Feb 2021)', color=INK,
                   fontsize=11, fontweight='bold')
ax_left.set_ylim(0, 35)
# Annotate 2021 specially
ax_left.annotate('Current model:\nus_real10y = -0.32%\n-> gold activates at 8.1%',
                  xy=(0, m0_gold[0]), xytext=(0.5, 18),
                  arrowprops=dict(arrowstyle='->', color='#C00000', lw=1),
                  fontsize=8, color='#C00000',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF5F5', edgecolor='#C00000', lw=0.5))
add_grid(ax_left)

# Right panel: mean gold in Q1 MALA scenarios
q1_data = df[df['question_id'] == 'Q1_gold_favorable']
for a_idx, anchor in enumerate(ANCHORS):
    asub = q1_data[q1_data['anchor_date'] == anchor]
    if asub.empty:
        continue
    if 'w_ALT_GLD' not in asub.columns:
        continue
    mean_gld = asub['w_ALT_GLD'].mean() * 100
    std_gld  = asub['w_ALT_GLD'].std()  * 100
    color    = ANCHOR_COLORS[anchor]
    ax_right.bar(a_idx, mean_gld, color=color, alpha=0.75, width=0.5, zorder=3)
    ax_right.errorbar(a_idx, mean_gld, yerr=std_gld, fmt='none',
                       color=INK, lw=1.5, capsize=5, zorder=4)
    ax_right.text(a_idx, mean_gld + std_gld + 0.5, f'{mean_gld:.1f}%',
                   ha='center', fontsize=9, fontweight='bold', color=INK)
    # m0 reference line per anchor
    ax_right.plot([a_idx-0.35, a_idx+0.35],
                   [M0[anchor]['w_ALT_GLD']*100]*2,
                   color=INK, lw=1.5, linestyle=':', zorder=5)

ax_right.set_xticks(range(4))
ax_right.set_xticklabels([ANCHOR_LABEL[a] for a in ANCHORS], fontsize=8)
ax_right.set_ylabel('Mean gold allocation in scenarios (%)', color=GREY)
ax_right.set_title('Gold Weight in Gold-Favorable Scenarios\n(MALA Layer)', color=INK,
                    fontsize=11, fontweight='bold')
ax_right.set_ylim(0, 35)
ax_right.text(0.5, 0.02, 'Dotted line = m0 reference weight',
               ha='center', transform=ax_right.transAxes, fontsize=8, color=GREY)
add_grid(ax_right)

# Central explanation text
fig.text(0.5, -0.04,
    "The 2021 anchor shows gold = 8.1% in the current model (negative real rates activate gold).\n"
    "The historical benchmark chart shows ~0% gold in 2021 because the rolling model used at that time had different training data and EN coefficients. Both are correct.",
    ha='center', fontsize=9, color=INK, style='italic', transform=fig.transFigure)
save(fig, 'fig09_benchmark_vs_mala')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("Figure inventory -- workspace_v4/reports/figures_v3/")
names = [
    'fig01_question_regime_compass',
    'fig02_macro_fingerprint',
    'fig03_mala_convergence',
    'fig04a_portfolio_q1_gold',
    'fig04b_portfolio_q2_ew',
    'fig04c_portfolio_q3_return',
    'fig05_rf_story',
    'fig06_macro_shift',
    'fig07_g_landscape',
    'fig08_dominant_scenario',
    'fig09_benchmark_vs_mala',
]
for n in names:
    png = FIGURES / f'{n}.png'
    pdf = FIGURES / f'{n}.pdf'
    status = 'OK' if (png.exists() and pdf.exists()) else 'MISSING'
    print(f'  {n:<40} {status}')
