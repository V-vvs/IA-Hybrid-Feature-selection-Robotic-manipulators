import argparse, json, warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent

def find_file(name):
    for root in [SCRIPT_DIR, Path.cwd(), SCRIPT_DIR.parent, Path.cwd().parent]:
        for m in sorted(root.rglob(name)):
            return m
    return None

def _find_output_dir():
    # Detect project root
    cwd = Path.cwd().resolve()
    for folder in [cwd, cwd.parent, cwd.parent.parent]:
        if (folder / 'src').exists():
            return folder / 'Validation_Scenarios'
    # Fallback
    return SCRIPT_DIR.parent / 'Validation_Scenarios'

OUT_DIR = Path(args.output).resolve() if args.output else _find_output_dir()
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output: {OUT_DIR}")

# ── Load Sc3 features ─────────────────────────────────────────────────────────
sc3_files = {
    'KNN':      'selected_features_cor80_cum95_bestKNN.json',
    'SVM':      'selected_features_cor90_cum80_bestSVM.json',
    'MLP':      'selected_features_cor95_cum80_bestMLP.json',
    'LightGBM': 'selected_features_cor75_cum85_bestLGBM.json',
}
sc3_meta = {
    'KNN':      ('0.80', '0.95'),
    'SVM':      ('0.90', '0.80'),
    'MLP':      ('0.95', '0.80'),
    'LightGBM': ('0.75', '0.85'),
}
sc3 = {}
for model, fname in sc3_files.items():
    p = find_file(fname)
    if p is None:
        raise FileNotFoundError(f"Not found: {fname}")
    sc3[model] = json.load(open(p))['features']

# ── Load Sc4 features (SHAP top-N at best threshold) ─────────────────────────
shap_path = find_file('shap_feature_rankings.csv')
if shap_path is None:
    raise FileNotFoundError("shap_feature_rankings.csv not found")
shap_df = pd.read_csv(shap_path)

# Best Sc4 config per model: (threshold%, n_features)
sc4_best = {
    'KNN':      ('85%', 67),
    'SVM':      ('85%', 34),
    'MLP':      ('85%', 31),
    'LightGBM': ('85%', 51),
}
sc4 = {}
for model, (thresh, n) in sc4_best.items():
    sc4[model] = (shap_df[shap_df['Model'] == model]
                  .sort_values('Rank').head(n)['Feature'].tolist())

# SHAP importance column
if 'SHAP_Importance_Overall' in shap_df.columns:
    shap_df['SHAP_Importance'] = shap_df['SHAP_Importance_Overall']

# ── Colours ───────────────────────────────────────────────────────────────────
C_BOTH  = '#2c6fad'   # blue  — both Sc3 & Sc4
C_SC4   = '#27ae60'   # green — only Sc4
C_SC3   = '#c0392b'   # red   — only Sc3
C_NONE  = '#95a5a6'   # grey  — neither

# ── Global style ──────────────────────────────────────────────────────────────
FS = dict(
    tick    = 30,
    label   = 32,
    title   = 34,
    suptitle= 38,
    annot   = 28,
    legend  = 34,  
    footer  = 26,
)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times New Roman', 'Times'],
    'text.usetex':       False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Normalise SHAP importance per model (0–1) ─────────────────────────────────
def get_shap_scores(model):
    sub = shap_df[shap_df['Model'] == model].copy()
    sub = sub.sort_values('Rank').reset_index(drop=True)
    mx  = sub['SHAP_Importance'].max()
    sub['norm'] = sub['SHAP_Importance'] / mx if mx > 0 else 0.0
    return dict(zip(sub['Feature'], sub['norm'])), sub['Feature'].tolist()

# ── Plot one figure per model ─────────────────────────────────────────────────
models = ['KNN', 'SVM', 'MLP', 'LightGBM']

for model in models:
    print(f"\nGenerating figure for {model}...")

    sc3_feats = sc3[model]           # ordered by ReliefF+ANOVA rank
    sc4_feats = sc4[model]           # ordered by SHAP rank
    sc3_set   = set(sc3_feats)
    sc4_set   = set(sc4_feats)

    shap_scores, all_shap_ranked = get_shap_scores(model)

    # Features selected by neither — sorted by global SHAP rank
    neither = [f for f in all_shap_ranked
               if f not in sc3_set and f not in sc4_set]

    # Colour assignment
    def bar_color(feat, ref_set_self, ref_set_other):
        in_self  = feat in ref_set_self
        in_other = feat in ref_set_other
        if in_self and in_other: return C_BOTH
        if in_self:              return C_SC4   # only in this panel's scenario
        return C_SC3                            # only in other scenario

    # Sc4 panel: ref_self=sc4_set, ref_other=sc3_set
    # green = only sc4, red = only sc3
    def color_sc4(feat):
        if feat in sc4_set and feat in sc3_set: return C_BOTH
        if feat in sc4_set:                     return C_SC4
        return C_SC3  

    def color_sc3(feat):
        if feat in sc3_set and feat in sc4_set: return C_BOTH
        if feat in sc3_set:                     return C_SC3
        return C_SC4

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_sc4    = len(sc4_feats)
    n_sc3    = len(sc3_feats)
    n_none   = len(neither)
    max_rows = max(n_sc4, n_sc3, n_none)

    # Height proportional to max features, min 16 inches
    fig_h = max(16, max_rows * 0.45)
    fig_w = 52   

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 1, 1],
        wspace=0.55,
        left=0.10, right=0.97,
        top=0.93,  bottom=0.13,
    )

    corr_l, cum_l = sc3_meta[model]
    thresh_l, n4  = sc4_best[model]

    ax_sc4   = fig.add_subplot(gs[0])
    ax_sc3   = fig.add_subplot(gs[1])
    ax_none  = fig.add_subplot(gs[2])

    # ── Panel left: Sc4 (SHAP ranked) ────────────────────────────────────────
    vals4   = [shap_scores.get(f, 0.0) for f in sc4_feats]
    colors4 = [color_sc4(f) for f in sc4_feats]

    ax_sc4.barh(range(n_sc4), vals4, color=colors4, height=0.75,
                alpha=0.90, zorder=3)
    ax_sc4.set_yticks(range(n_sc4))
    ax_sc4.set_yticklabels(sc4_feats, fontsize=FS['tick'],
                            fontfamily='monospace')
    ax_sc4.invert_yaxis()
    ax_sc4.set_xlabel('Normalised SHAP Importance', fontsize=FS['label'], labelpad=10)
    ax_sc4.set_xlim(0, 1.05)
    ax_sc4.set_title(
        f'Scenario 4 — SHAP Ranking\nSHAP Threshold={thresh_l} | {n4} feat.',
        fontsize=FS['title'], fontweight='bold', pad=14
    )
    ax_sc4.xaxis.set_tick_params(labelsize=FS['tick'])
    ax_sc4.grid(axis='x', linestyle=':', alpha=0.35, zorder=0)

    # ── Panel center: Sc3 (ReliefF+ANOVA ranked) ──────────────────────────────
    vals3   = [shap_scores.get(f, 0.0) for f in sc3_feats]
    colors3 = [color_sc3(f) for f in sc3_feats]

    ax_sc3.barh(range(n_sc3), vals3, color=colors3, height=0.75,
                alpha=0.90, zorder=3)
    ax_sc3.set_yticks(range(n_sc3))
    ax_sc3.set_yticklabels(sc3_feats, fontsize=FS['tick'],
                            fontfamily='monospace')
    ax_sc3.invert_yaxis()
    ax_sc3.set_xlabel('Normalised SHAP Importance', fontsize=FS['label'], labelpad=10)
    ax_sc3.set_xlim(0, 1.05)
    ax_sc3.set_title(
        f'Scenario 3\nCorr={corr_l}, Cum={cum_l} | {n_sc3} feat.',
        fontsize=FS['title'], fontweight='bold', pad=14
    )
    ax_sc3.xaxis.set_tick_params(labelsize=FS['tick'])
    ax_sc3.grid(axis='x', linestyle=':', alpha=0.35, zorder=0)

    # ── Panel right: neither ──────────────────────────────────────────────────
    vals_none   = [shap_scores.get(f, 0.0) for f in neither]
    colors_none = [C_NONE] * len(neither)

    ax_none.barh(range(n_none), vals_none, color=colors_none, height=0.75,
                 alpha=0.85, zorder=3)
    # Smaller font for panel right
    tick_none = max(14, FS['tick'] - 12)
    ax_none.set_yticks(range(n_none))
    ax_none.set_yticklabels(neither, fontsize=tick_none,
                             fontfamily='monospace')
    ax_none.invert_yaxis()
    ax_none.set_xlabel('Normalised SHAP Importance', fontsize=FS['label'], labelpad=10)
    ax_none.set_xlim(0, 1.05)
    ax_none.set_title(
        'Selected by Neither Method\nsorted by overall SHAP rank',
        fontsize=FS['title'], fontweight='bold', pad=14
    )
    ax_none.xaxis.set_tick_params(labelsize=FS['tick'])
    ax_none.grid(axis='x', linestyle=':', alpha=0.35, zorder=0)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f'{model} — Feature Selection Comparison: Scenario 3 vs. Scenario 4',
        fontsize=FS['suptitle'], fontweight='bold', y=0.975
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=C_BOTH, label='Selected by Sc3 & Sc4'),
        mpatches.Patch(color=C_SC4,  label='Selected only by Sc4'),
        mpatches.Patch(color=C_SC3,  label='Selected only by Sc3'),
        mpatches.Patch(color=C_NONE, label='Selected by neither'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=4,
        fontsize=FS['legend'],
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
        handleheight=2.0,
        handlelength=3.0,
        borderpad=1.2,
        labelspacing=1.0,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out = OUT_DIR / f'fig_feature_overlap_{model}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out}")

print("\nDone.")
