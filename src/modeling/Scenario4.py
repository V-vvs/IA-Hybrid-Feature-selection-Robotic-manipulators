
import argparse, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input',  type=str, default=None)
parser.add_argument('--shap',   type=str, default=None)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent

def find_file(name):
    candidates = [Path.cwd() / name, SCRIPT_DIR / name]
    for root in [SCRIPT_DIR, Path.cwd()]:
        r = root
        for _ in range(4):
            for m in sorted(r.rglob(name)):
                if m not in candidates:
                    candidates.append(m)
            r = r.parent
    return next((c for c in candidates if c.exists()), None)

csv_path  = Path(args.input).resolve() if args.input  else find_file('model_specific_optuna_complete_summary.csv')
shap_path = Path(args.shap).resolve()  if args.shap   else find_file('shap_feature_rankings.csv')

if csv_path is None:
    print("ERROR: model_specific_optuna_complete_summary.csv not found."); sys.exit(1)
if shap_path is None:
    print("ERROR: shap_feature_rankings.csv not found."); sys.exit(1)

OUT_DIR = Path(args.output).resolve() if args.output else csv_path.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Summary  : {csv_path}")
print(f"SHAP     : {shap_path}")
print(f"Output   : {OUT_DIR}")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
df['Threshold_num'] = df['Threshold'].str.replace('%','').astype(int)
shap_df = pd.read_csv(shap_path)
if 'SHAP_Importance_Overall' in shap_df.columns:
    shap_df['SHAP_Importance'] = shap_df['SHAP_Importance_Overall']

models        = ['KNN','SVM','MLP','LightGBM']
thresh_order  = [80, 85, 90, 95]
thresh_labels = ['80%','85%','90%','95%']

# Best config per model
best_per_model = df.loc[df.groupby('Model')['F1_Macro'].idxmax()].set_index('Model')

cmap_blue = LinearSegmentedColormap.from_list(
    'purple', ['#fcfbfd','#dadaeb','#9e9ac8','#6a51a3','#3f007d'])
vmin, vmax = 0.84, 1.00

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: draw heatmap axes
# ══════════════════════════════════════════════════════════════════════════════
def draw_heatmap(axes, fontsize_annot=9.5, fontsize_tick=11,
                 fontsize_ylabel=12, fontsize_title=13, show_cbar_on=3):
    for col_i, (model, ax) in enumerate(zip(models, axes)):
        sub = df[df['Model']==model].copy()
        pivot_f1   = sub.set_index('Threshold_num')['F1_Macro'].reindex(thresh_order)
        pivot_feat = sub.set_index('Threshold_num')['Num_Features'].reindex(thresh_order)
        vals  = pivot_f1.values.reshape(-1, 1)
        feats = pivot_feat.values.reshape(-1, 1)

        im = ax.imshow(vals, aspect='auto', cmap=cmap_blue,
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        for r in range(len(thresh_order)):
            v = vals[r, 0]
            n = feats[r, 0]
            if not np.isnan(v):
                brightness = (v - vmin) / (vmax - vmin)
                color = 'white' if brightness > 0.60 else '#1a1a1a'
                ax.text(0, r, f'{v:.3f}\n({int(n)})',
                        ha='center', va='center', fontsize=fontsize_annot,
                        color=color, linespacing=1.4)

        # mark best row
        best_thresh = int(best_per_model.loc[model, 'Threshold_num'])
        best_row    = thresh_order.index(best_thresh)
        for spine in ['top','bottom','left','right']:
            ax.spines[spine].set_visible(False)
        rect = plt.Rectangle((-0.5, best_row - 0.5), 1, 1,
                              linewidth=2.5, edgecolor='#e63900',
                              facecolor='none', zorder=5)
        ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks(range(len(thresh_order)))
        if col_i == 0:
            ax.set_yticklabels(thresh_labels, fontsize=fontsize_tick)
            ax.set_ylabel('SHAP Cumulative Threshold', fontsize=fontsize_ylabel)
        else:
            ax.set_yticklabels([])

        ax.set_title(model, fontsize=fontsize_title, fontweight='bold', pad=6)
        ax.set_yticks(np.arange(len(thresh_order)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.0)
        ax.tick_params(which='minor', left=False)

        if col_i == show_cbar_on:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=fontsize_tick)
            cbar.set_label('F1-Macro', fontsize=fontsize_ylabel)
            cbar.set_ticks([0.84, 0.88, 0.92, 0.96, 1.00])
    return im

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — standalone heatmap
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({'font.family':'serif',
                     'font.serif':['DejaVu Serif','Times New Roman','Times'],
                     'text.usetex':False})

fig1, axes1 = plt.subplots(1, 4, figsize=(16, 5.5),
                            gridspec_kw={'wspace':0.06,
                                         'left':0.08,'right':0.93,
                                         'top':0.85,'bottom':0.12})
fig1.suptitle('Scenario 4 — F1-Macro per Model and SHAP Cumulative Threshold',
              fontsize=13, fontweight='bold', y=0.97)
draw_heatmap(axes1)
fig1.text(0.5, 0.02,
          'Cell annotation: F1-Macro score; (n) = number of selected features. '
          '',
          ha='center', fontsize=9, style='italic', color='#444')
out1 = OUT_DIR / 'fig_sc4_heatmap.png'
plt.savefig(out1, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — heatmap + aggregated sensitivity
# ══════════════════════════════════════════════════════════════════════════════
model_colors = {'KNN':'#1f77b4','SVM':'#d62728','MLP':'#2ca02c','LightGBM':'#ff7f0e'}

FS = dict(annot=40, tick=43, ylabel=46, title=49, legend=40, suptitle=52)

fig2 = plt.figure(figsize=(52, 36))
gs   = gridspec.GridSpec(2, 4, figure=fig2,
                         height_ratios=[1, 1.4],
                         hspace=0.22,         
                         wspace=0.30,
                         left=0.07, right=0.97,
                         top=0.93, bottom=0.09)

# ── Row 0: heatmap ────────────────────────────────────────────────────────────
hm_axes = [fig2.add_subplot(gs[0, i]) for i in range(4)]
draw_heatmap(hm_axes,
             fontsize_annot=FS['annot'], fontsize_tick=FS['tick'],
             fontsize_ylabel=FS['ylabel'], fontsize_title=FS['title'])
fig2.text(0.50, 0.962,
          'Scenario 4 — F1-Macro Heatmap and Aggregated Sensitivity Analysis',
          ha='center', fontsize=FS['suptitle'], fontweight='bold')
fig2.text(0.50, 0.944,
          '',
          ha='center', fontsize=FS['legend'], style='italic', color='#e63900')

# ── Row 1 left: F1 vs Threshold per model ─────────────────────────────────────
ax_line = fig2.add_subplot(gs[1, :2])
for model in models:
    sub = df[df['Model']==model].sort_values('Threshold_num')
    ax_line.plot(sub['Threshold_num'], sub['F1_Macro'],
                 marker='o', markersize=18, linewidth=4.5,
                 color=model_colors[model], label=model)
    ax_line.fill_between(sub['Threshold_num'],
                         sub['F1_Macro'] - 0.005,
                         sub['F1_Macro'] + 0.005,
                         alpha=0.08, color=model_colors[model])

ax_line.set_xticks(thresh_order)
ax_line.set_xticklabels(thresh_labels, fontsize=FS['tick'])
ax_line.set_xlabel('SHAP Cumulative Threshold', fontsize=FS['ylabel'])
ax_line.set_ylabel('F1-Macro', fontsize=FS['ylabel'])
ax_line.set_title('F1-Macro per Model vs Threshold', fontsize=FS['title'], fontweight='bold')
ax_line.set_ylim(0.84, 1.02)
ax_line.yaxis.set_tick_params(labelsize=FS['tick'])
ax_line.grid(True, linestyle=':', alpha=0.4)
ax_line.legend(fontsize=FS['legend'], loc='lower right', framealpha=0.9)

# ── Row 1 right: aggregated mean ± std ────────────────────────────────────────
ax_agg = fig2.add_subplot(gs[1, 2:])
agg = df.groupby('Threshold_num')['F1_Macro'].agg(['mean','std']).reindex(thresh_order)
ax_agg.plot(thresh_order, agg['mean'], marker='s', markersize=18,
            linewidth=4.5, color='#3f007d', label='Mean across models')
ax_agg.fill_between(thresh_order,
                    agg['mean'] - agg['std'],
                    agg['mean'] + agg['std'],
                    alpha=0.18, color='#3f007d', label='± std')

ax_agg.set_xticks(thresh_order)
ax_agg.set_xticklabels(thresh_labels, fontsize=FS['tick'])
ax_agg.set_xlabel('SHAP Cumulative Threshold', fontsize=FS['ylabel'])
ax_agg.set_ylabel('F1-Macro', fontsize=FS['ylabel'])
ax_agg.set_title('Aggregated F1-Macro\n(mean ± std across models)',
                 fontsize=FS['title'], fontweight='bold')
ax_agg.set_ylim(0.84, 1.02)
ax_agg.yaxis.set_tick_params(labelsize=FS['tick'])
ax_agg.grid(True, linestyle=':', alpha=0.4)
ax_agg.legend(fontsize=FS['legend'], framealpha=0.9)

out2 = OUT_DIR / 'fig_sc4_combined.png'
plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — horizontal stacked bars per class, 2x2 subplots
# ══════════════════════════════════════════════════════════════════════════════
class_cols   = ['SHAP_collision','SHAP_normal','SHAP_obstruction']
class_labels = ['collision','normal','obstruction']
class_colors = ['#d62728','#2ca02c','#ff7f0e']   # red, green, orange

fig3, axes3 = plt.subplots(2, 2, figsize=(32, 38))
fig3.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.05,
                     hspace=0.30, wspace=0.45)
fig3.suptitle('Scenario 4 — SHAP Feature Importance per Class\n(Best Configuration per Model)',
              fontsize=20, fontweight='bold')

for ax, model in zip(axes3.flat, models):
    best_thresh_num = int(best_per_model.loc[model, 'Threshold_num'])
    n_feat          = int(best_per_model.loc[model, 'Num_Features'])
    f1_val          = float(best_per_model.loc[model, 'F1_Macro'])

    model_shap = (shap_df[shap_df['Model']==model]
                  .sort_values('Rank')
                  .head(n_feat)
                  .reset_index(drop=True))

    # Normalize each class column by its own max so bars are comparable
    for col in class_cols:
        mx = model_shap[col].max()
        model_shap[col+'_norm'] = model_shap[col] / mx if mx > 0 else 0

    norm_cols = [c+'_norm' for c in class_cols]

    features = model_shap['Feature'].tolist()
    y        = np.arange(len(features))

    # Stacked horizontal bars — sorted by overall importance (rank 1 on top)
    left = np.zeros(len(features))
    for col, label, color in zip(norm_cols, class_labels, class_colors):
        vals = model_shap[col].values
        ax.barh(y, vals, left=left, color=color, label=label, alpha=0.88, height=0.7)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()   # rank 1 at top
    ax.set_xlabel('Cumulative Normalized SHAP Importance', fontsize=13)
    ax.set_title(
        f'{model}  |  Threshold={best_thresh_num}%  |  {n_feat} feat.  |  F1={f1_val:.3f}',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

out3 = OUT_DIR / 'fig_sc4_feature_ranking.png'
plt.savefig(out3, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out3}")
