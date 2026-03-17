
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

# ── Path resolution ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input',  type=str, default=None)
parser.add_argument('--folder', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent

if args.input:
    csv_path = Path(args.input).resolve()
elif args.folder:
    csv_path = (Path(args.folder) / 'grid_search_summary.csv').resolve()
else:
    candidates = [Path.cwd() / 'grid_search_summary.csv',
                  SCRIPT_DIR / 'grid_search_summary.csv']
    for search_root in [SCRIPT_DIR, Path.cwd()]:
        root = search_root
        for _ in range(4):
            for match in sorted(root.rglob('grid_search_summary.csv')):
                if match not in candidates:
                    candidates.append(match)
            root = root.parent
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        print("ERROR: grid_search_summary.csv not found.")
        sys.exit(1)

OUT_DIR = Path(args.output).resolve() if args.output else csv_path.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Input : {csv_path}\nOutput: {OUT_DIR}")

# ── Matplotlib base settings ────────────────────────
plt.rcParams.update({
    'font.family':           'serif',
    'font.serif':            ['DejaVu Serif', 'Times New Roman', 'Times'],
    'font.size':             26,
    'axes.titlesize':        30,
    'axes.labelsize':        26,
    'xtick.labelsize':       24,
    'ytick.labelsize':       24,
    'legend.fontsize':       24,
    'legend.title_fontsize': 24,
    'axes.linewidth':        0.8,
    'text.usetex':           False,
})

# ── Load & prepare ─────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)

def corr_label(v):
    try:    return str(round(float(v), 2))
    except: return 'NoCorr'

df['Corr_Label'] = df['Corr_Threshold'].apply(corr_label)

corr_order = ['0.7','0.75','0.8','0.85','0.9','0.95','1.0','NoCorr']
cum_order  = [0.80, 0.85, 0.90, 0.95, 1.00]
cum_labels = ['0.80','0.85','0.90','0.95','1.00']
models     = ['KNN','SVM','MLP','LightGBM']

cmap_blue = LinearSegmentedColormap.from_list(
    'blue', ['#f7fbff','#c6dbef','#6baed6','#2171b5','#084594'])

vmin, vmax = 0.84, 1.00

# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP FIGURES — layout 2×2
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(metric, metric_label, filename):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(32, 30),
        gridspec_kw={
            'hspace': 0.15,
            'wspace': 0.10,
            'left':   0.11,
            'right':  0.87,
            'top':    0.94,
            'bottom': 0.08,
        }
    )

    fig.suptitle(
        f'Grid Search Results — {metric_label} per Model and Threshold Combination',
        fontsize=34, fontweight='bold', y=0.978
    )

    axes_flat = axes.flatten()   # KNN, SVM, MLP, LightGBM

    for col_i, (model, ax) in enumerate(zip(models, axes_flat)):
        sub = df[df['Model'] == model].copy()
        sub['Corr_Label'] = sub['Corr_Threshold'].apply(corr_label)

        pivot_val  = sub.pivot_table(index='Corr_Label', columns='Cum_Threshold',
                                     values=metric, aggfunc='mean')
        pivot_feat = sub.pivot_table(index='Corr_Label', columns='Cum_Threshold',
                                     values='Num_Features', aggfunc='mean')

        pivot_val  = pivot_val.reindex(corr_order).reindex(columns=cum_order)
        pivot_feat = pivot_feat.reindex(corr_order).reindex(columns=cum_order)

        im = ax.imshow(pivot_val.values, aspect='auto', cmap=cmap_blue,
                       vmin=vmin, vmax=vmax, interpolation='nearest')

        for r in range(pivot_val.shape[0]):
            for c in range(pivot_val.shape[1]):
                v = pivot_val.values[r, c]
                n = pivot_feat.values[r, c]
                if not np.isnan(v):
                    brightness = (v - vmin) / (vmax - vmin)
                    color = 'white' if brightness > 0.60 else '#1a1a1a'
                    ax.text(c, r, f'{v:.3f}\n({int(n)})',
                            ha='center', va='center', fontsize=24,
                            color=color, linespacing=1.35,
                            fontweight='bold')

        ax.set_xticks(range(len(cum_order)))
        ax.set_xticklabels(cum_labels, rotation=45, ha='right', fontsize=24)
        ax.set_yticks(range(len(corr_order)))

        # y axis
        if col_i % 2 == 0:
            ax.set_yticklabels(corr_order, fontsize=24)
            ax.set_ylabel('Pearson Correlation Threshold ($\\rho$)', fontsize=26,
                          labelpad=10)
        else:
            ax.set_yticklabels([])

        # x axis
        if col_i >= 2:
            ax.set_xlabel(r'Cumulative Threshold ($\tau_{cum}$)', fontsize=26,
                          labelpad=10)

        ax.set_title(model, fontsize=30, fontweight='bold', pad=12)

        # Grid minor
        ax.set_xticks(np.arange(len(cum_order))  - 0.5, minor=True)
        ax.set_yticks(np.arange(len(corr_order)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', bottom=False, left=False)

        # global
        if col_i == 3:
            im_last = im

    #  global Colorbar
    cbar_ax = fig.add_axes([0.89, 0.08, 0.018, 0.86])   # [left, bottom, width, height]
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=26)
    cbar.set_label(metric_label, fontsize=26, labelpad=14)
    cbar.set_ticks([0.84, 0.88, 0.92, 0.96, 1.00])

    fig.text(
        0.50, 0.02,
        'Cell annotation: upper value = metric score; '
        'lower value (in parentheses) = number of selected features',
        ha='center', fontsize=26, style='italic', color='#444444'
    )

    out = OUT_DIR / filename
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


make_figure('Accuracy', 'Accuracy', 'fig_journal_accuracy.png')
make_figure('F1_Macro', 'F1-Macro', 'fig_journal_f1macro.png')


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY FIGURE — F1-Macro vs Corr Threshold  +  F1-Macro vs Cum Threshold
# ══════════════════════════════════════════════════════════════════════════════
def make_sensitivity_figure():
    print("Creating sensitivity figure...")

    cum_colors  = {
        0.80: '#08306b', 0.85: '#2171b5',
        0.90: '#6baed6', 0.95: '#41ab5d', 1.00: '#addd8e',
    }
    corr_colors = {
        '0.7':    '#3f007d', '0.75': '#756bb1', '0.8':  '#d6604d',
        '0.85':   '#f4a582', '0.9':  '#d95f02', '0.95': '#fdae6b',
        '1.0':    '#8c510a', 'NoCorr': '#f6e8c3',
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(28, 9),
        gridspec_kw={'left': 0.07, 'right': 0.95, 'top': 0.88,
                     'bottom': 0.16, 'wspace': 0.32}
    )

    # ── helpers ───────────────────────────────────────────────────────────────
    def corr_num(v):
        try:    return float(v)
        except: return np.nan   # NoCorr

    corr_num_order = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, np.nan]
    corr_x_labels  = ['0.7','0.75','0.8','0.85','0.9','0.95','1.0','NoCorr']
    corr_x_pos     = list(range(len(corr_x_labels)))

    # ── LEFT: F1 vs Corr Threshold ──────
    for cum in cum_order:
        sub  = df[df['Cum_Threshold'] == cum].copy()
        sub['Corr_x'] = sub['Corr_Label'].apply(
            lambda v: corr_x_labels.index(v) if v in corr_x_labels else None)
        sub = sub.dropna(subset=['Corr_x'])
        agg  = (sub.groupby('Corr_x')['F1_Macro']
                   .agg(['mean','std'])
                   .reindex(corr_x_pos))
        color = cum_colors[cum]
        ax1.plot(corr_x_pos, agg['mean'], marker='o', markersize=9,
                 linewidth=2.5, color=color, label=f'Cum={cum:.2f}')
        ax1.fill_between(corr_x_pos,
                         agg['mean'] - agg['std'],
                         agg['mean'] + agg['std'],
                         alpha=0.18, color=color)

    ax1.set_xticks(corr_x_pos)
    ax1.set_xticklabels(corr_x_labels, fontsize=24)
    ax1.set_xlabel('Correlation Threshold', fontsize=26)
    ax1.set_ylabel('F1-Macro', fontsize=26)
    ax1.set_ylim(0.800, 1.005)
    ax1.yaxis.set_tick_params(labelsize=24)
    ax1.set_title('F1-Macro vs Corr Threshold\n(mean ± std over models)',
                  fontsize=28, fontweight='bold')
    ax1.legend(fontsize=22, loc='lower right', framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.4)

    # ── RIGHT: F1 vs Cum Threshold ─────
    for corr_lbl in corr_order:
        sub = df[df['Corr_Label'] == corr_lbl].copy()
        agg = (sub.groupby('Cum_Threshold')['F1_Macro']
                  .agg(['mean','std'])
                  .reindex(cum_order))
        color = corr_colors[corr_lbl]
        ax2.plot(cum_order, agg['mean'], marker='o', markersize=9,
                 linewidth=2.5, color=color, label=f'Corr={corr_lbl}')
        ax2.fill_between(cum_order,
                         agg['mean'] - agg['std'],
                         agg['mean'] + agg['std'],
                         alpha=0.18, color=color)

    ax2.set_xticks(cum_order)
    ax2.set_xticklabels([f'{c:.2f}' for c in cum_order], fontsize=24)
    ax2.set_xlabel('Cumulative Threshold', fontsize=26)
    ax2.set_ylabel('F1-Macro', fontsize=26)
    ax2.set_ylim(0.800, 1.005)
    ax2.yaxis.set_tick_params(labelsize=24)
    ax2.set_title('F1-Macro vs Cum Threshold\n(mean ± std over models)',
                  fontsize=28, fontweight='bold')
    ax2.legend(fontsize=22, loc='lower right', framealpha=0.9,
               ncol=1)
    ax2.grid(True, linestyle=':', alpha=0.4)

    out = OUT_DIR / 'fig_sensitivity_combined.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


make_sensitivity_figure()



# ══════════════════════════════════════════════════════════════════════════════
# CONFUSION MATRIX FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def make_confusion_figure():
    """
    Lê grid_search_summary.csv para encontrar a melhor configuração por modelo
    (maior F1_Macro, desempate: menor Num_Features) e os arquivos de predição
    correspondentes para montar as matrizes de confusão.

    Espera encontrar, para cada combo vencedor, um arquivo:
      <scenario3_folder>/<Combo_Label>/predictions_<Model>_test.csv
    com colunas: y_true, y_pred

    Se os arquivos não existirem, tenta usar colunas do próprio summary.
    """
    from sklearn.metrics import confusion_matrix
    import os

    print("Creating confusion matrix figure...")

    classes = ['collision', 'normal', 'obstruction']
    cmap_cm = LinearSegmentedColormap.from_list(
        'cm_blue', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#084594'])

    # ── best config ──────────────────────────────────────────────
    df_s = pd.read_csv(csv_path)
    df_s['F1_Macro']     = pd.to_numeric(df_s['F1_Macro'],     errors='coerce')
    df_s['Num_Features'] = pd.to_numeric(df_s['Num_Features'], errors='coerce')

    def corr_label_local(v):
        try:    return str(round(float(v), 2))
        except: return 'NoCorr'

    df_s['Corr_Label'] = df_s['Corr_Threshold'].apply(corr_label_local)

    best_rows = (df_s.sort_values(['F1_Macro', 'Num_Features'],
                                   ascending=[False, True])
                     .groupby('Model', sort=False)
                     .first()
                     .reset_index())

    sc3_dir = csv_path.parent   # path - grid_search_summary.csv

    fig, axes = plt.subplots(
        2, 2,
        figsize=(22, 20),
        gridspec_kw={
            'hspace': 0.35,
            'wspace': 0.35,
            'left':   0.08,
            'right':  0.96,
            'top':    0.93,
            'bottom': 0.07,
        }
    )
    fig.suptitle('Confusion Matrices — Best Configuration per Model (Test Set)',
                 fontsize=30, fontweight='bold', y=0.975)

    axes_flat = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        row = best_rows[best_rows['Model'] == model].iloc[0]

        combo   = row['Combo_Label']
        corr_l  = row['Corr_Label']
        cum_t   = row['Cum_Threshold']
        n_feat  = int(row['Num_Features'])
        f1_val  = row['F1_Macro']

        # prediction files
        pred_file = sc3_dir / combo / f'predictions_{model}_test.csv'
        if not pred_file.exists():
            found = list(sc3_dir.rglob(f'predictions_{model}_test.csv'))
            if found:
                pred_file = found[0]

        if pred_file.exists():
            preds  = pd.read_csv(pred_file)
            y_true = preds['y_true'].tolist()
            y_pred = preds['y_pred'].tolist()
            acc    = np.mean(np.array(y_true) == np.array(y_pred))
        else:
            # fallback
            print(f"   WARNING: predictions file not found for {model} — using summary Accuracy")
            acc = row.get('Accuracy', float('nan'))
            y_true, y_pred = [], []

        if y_true:
            cm = confusion_matrix(y_true, y_pred, labels=classes)
        else:
            cm = np.full((3, 3), np.nan)

        # Plot
        im = ax.imshow(cm, cmap=cmap_cm, aspect='auto', interpolation='nearest')

        for r in range(len(classes)):
            for c in range(len(classes)):
                val = cm[r, c]
                if not np.isnan(val):
                    vmax_cm = cm.max() if cm.max() > 0 else 1
                    brightness = val / vmax_cm
                    color = 'white' if brightness > 0.55 else '#1a1a1a'
                    ax.text(c, r, str(int(val)),
                            ha='center', va='center',
                            fontsize=28, fontweight='bold', color=color)

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=22)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=22)
        ax.set_xlabel('Predicted', fontsize=24, labelpad=8)
        ax.set_ylabel('Actual',    fontsize=24, labelpad=8)

        title_line1 = model
        title_line2 = f'Acc={acc:.3f}  F1={f1_val:.3f}'
        title_line3 = f'rho={corr_l}  tau_cum={cum_t:.2f}  ({n_feat} feat.)'
        ax.set_title(f'{title_line1}\n{title_line2}\n{title_line3}',
                     fontsize=24, fontweight='bold', pad=10)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=20)

        # Grid minor
        ax.set_xticks(np.arange(len(classes)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(classes)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', bottom=False, left=False)

    out = OUT_DIR / 'fig_confusion_best.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


make_confusion_figure()


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE RANKING FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def make_ranking_figure():
    scores_path = csv_path.parent / "global_feature_scores.csv"

    if not scores_path.exists():
        for search_root in [SCRIPT_DIR, Path.cwd()]:
            root = search_root
            for _ in range(4):
                for match in sorted(root.rglob("global_feature_scores.csv")):
                    scores_path = match
                    break
                if scores_path.exists():
                    break
                root = root.parent
            if scores_path.exists():
                break

    if not scores_path.exists():
        print("WARNING: global_feature_scores.csv not found — skipping ranking figure.")
        return

    df_s = (pd.read_csv(scores_path)
              .sort_values("Combined_Score", ascending=False)
              .head(40)
              .reset_index(drop=True))
    df_s["ReliefF_norm"] = df_s["ReliefF_Score"] / df_s["ReliefF_Score"].max()
    df_s["ANOVA_norm"]   = df_s["ANOVA_Score"]   / df_s["ANOVA_Score"].max()

    features = df_s["Feature"].tolist()
    n = len(features)
    x = np.arange(n)
    w = 0.26

    fig, ax = plt.subplots(figsize=(24, 7))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.38)

    ax.bar(x - w, df_s["ReliefF_norm"],   width=w, color="#E8A020",
           label="ReliefF (norm)", zorder=3)
    ax.bar(x,     df_s["ANOVA_norm"],     width=w, color="#3A7D3A",
           label="ANOVA (norm)", zorder=3)
    ax.bar(x + w, df_s["Combined_Score"], width=w, color="#1B5FA8",
           label="Combined (norm)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=18)
    ax.set_ylabel("Normalized Score", fontsize=22)
    ax.set_xlabel("Features", fontsize=22)
    ax.set_title("Feature Importance — Top 40 Features\n"
                 "Combined ReliefF + ANOVA Ranking",
                 fontsize=24, fontweight="bold")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.legend(fontsize=18, loc="upper right", framealpha=0.9,
              edgecolor="#ccc", handlelength=1.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = OUT_DIR / "fig_journal_feature_ranking.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {out}")


make_ranking_figure()
