"""
UMAP GENERATOR — SCENARIOS 1, 2, 3
====================================================================
Sc1: raw sensor data flattened (Fx_1..Fz_N) — dataset_train/test_original.csv
Sc2: todas as 144 features transformadas   — dataset_train/test_transformed.csv
Sc3: features do combo corr×cumulative com maior F1_Macro individual;
     desempate por menor numero de features.
     Le grid_search_summary.csv e selected_features.json do combo vencedor.
"""

import ast
import json
import os
import warnings

import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ====================================================================
# CONFIGURATION
# ====================================================================

CONFIG = {
    'dataset_type': 'train',   # 'train' ou 'test'

    'umap_settings': {
        'n_neighbors':  25,
        'min_dist':     0.1,
        'n_components': 2,
        'random_state': 42,
    },

    'plot_settings': {
        'width':      900,
        'height':     700,
        'title_size': 40,   # dobrado (era 20)
        'font_size':  32,   # dobrado (era 16)
        'point_size': 8,
        'opacity':    0.7,
    },

    # Cores fixas por classe: normal=verde, collision=vermelho, obstruction=azul
    'class_colors': {
        'normal':      '#2ca02c',
        'collision':   '#d62728',
        'obstruction': '#1f77b4',
    },

    'folders': {
        'preprocessed': './Preprocessed_Data',
        'scenario3':    './ML_Results_Scenario3_GridSearch',
        'output':       './UMAP_Scenarios_1_2_3',
    },

    'sc1_variables': ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
}

# ====================================================================
# SCENARIO 1 — raw data flatten
# ====================================================================

def _flatten_raw_data(df, vars_list, max_len=None, is_training=True):
    for var in vars_list:
        df[var] = df[var].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    if is_training:
        current_max = 0
        for var in vars_list:
            valid = df[var].dropna()
            if not valid.empty:
                current_max = max(current_max, int(valid.apply(len).max()))
        max_len_use = current_max
    else:
        max_len_use = max_len

    flattened = []
    for _, row in df.iterrows():
        row_feat = {}
        for var in vars_list:
            vals   = np.array(row[var], dtype=float)
            padded = (np.full(max_len_use, np.nan)
                      if vals.size == 0 or np.all(np.isnan(vals))
                      else np.pad(vals, (0, max_len_use - vals.size),
                                  'constant', constant_values=np.nan))
            for i, v in enumerate(padded):
                row_feat[f'{var}_{i+1}'] = v
        flattened.append(row_feat)

    df_flat = pd.DataFrame(flattened, index=df.index)
    df_flat = df_flat.replace([np.inf, -np.inf], np.nan)
    if df_flat.isnull().values.any():
        medians = df_flat.median(numeric_only=True).fillna(0)
        df_flat = df_flat.fillna(medians).fillna(0)

    return df_flat, max_len_use


def load_scenario1(preprocessed_path, dataset_type, variables):
    print(f"\n{'='*70}")
    print(" LOADING SCENARIO 1: RAW SENSOR DATA (flatten)")
    print(f"{'='*70}")

    path = os.path.join(preprocessed_path, f'dataset_{dataset_type}_original.csv')
    if not os.path.exists(path):
        print(f"   ERROR: {path} not found!")
        return None, None

    df = pd.read_csv(path)
    print(f"   Raw shape: {df.shape}")

    X_flat, seq_len = _flatten_raw_data(df.copy(), variables, is_training=True)
    y = df['label'].reset_index(drop=True)

    print(f"   Flattened shape : {X_flat.shape}")
    print(f"   Sequence length : {seq_len}")
    print(f"   Classes         : {y.unique().tolist()}")
    return X_flat, y


# ====================================================================
# SCENARIO 2 — all transformed features
# ====================================================================

def load_scenario2(preprocessed_path, dataset_type):
    print(f"\n{'='*70}")
    print(" LOADING SCENARIO 2: ALL TRANSFORMED FEATURES (144)")
    print(f"{'='*70}")

    path = os.path.join(preprocessed_path, f'dataset_{dataset_type}_transformed.csv')
    if not os.path.exists(path):
        print(f"   ERROR: {path} not found!")
        return None, None

    df = pd.read_csv(path)
    X  = df.drop('label', axis=1)
    y  = df['label']

    print(f"   Shape    : {df.shape}")
    print(f"   Features : {X.shape[1]}")
    print(f"   Classes  : {y.unique().tolist()}")
    return X, y


# ====================================================================
# SCENARIO 3 — best combo by F1_Macro (tie-break: fewer features)
# ====================================================================

def find_best_combo_scenario3(scenario3_path):
    print(f"\n{'='*70}")
    print(" FINDING BEST COMBO FOR SCENARIO 3 (by F1_Macro, tie-break: fewer features)")
    print(f"{'='*70}")

    summary_file = os.path.join(scenario3_path, 'grid_search_summary.csv')
    if not os.path.exists(summary_file):
        print(f"   ERROR: {summary_file} not found!")
        return None, None, None

    df = pd.read_csv(summary_file)
    print(f"   Loaded grid_search_summary.csv: {df.shape}")

    df['F1_Macro']     = pd.to_numeric(df['F1_Macro'],     errors='coerce')
    df['Num_Features'] = pd.to_numeric(df['Num_Features'], errors='coerce')

    combo_best = (
        df.groupby(['Corr_Threshold', 'Cum_Threshold', 'Combo_Label', 'Num_Features'])
        ['F1_Macro'].max()
        .reset_index()
        .rename(columns={'F1_Macro': 'Best_F1_Macro'})
    )

    combo_best = combo_best.sort_values(
        ['Best_F1_Macro', 'Num_Features'],
        ascending=[False, True]
    ).reset_index(drop=True)

    best          = combo_best.iloc[0]
    combo_label   = best['Combo_Label']
    corr_label    = best['Corr_Threshold']
    cum_threshold = best['Cum_Threshold']
    best_acc      = best['Best_F1_Macro']
    n_features    = int(best['Num_Features'])

    print(f"\n   Best combo  : {combo_label}")
    print(f"   Corr        : {corr_label}  |  Cum: {cum_threshold:.2f}")
    print(f"   Best F1_Macro (individual model): {best_acc:.4f}")
    print(f"   Num features: {n_features}")

    best_row = df[(df['Combo_Label'] == combo_label) & (df['F1_Macro'] == best_acc)]
    if not best_row.empty:
        print(f"   Best model  : {best_row.iloc[0]['Model']}")

    print(f"\n   Top 5 combos:")
    print(combo_best[['Combo_Label', 'Best_F1_Macro', 'Num_Features']].head(5).to_string(index=False))

    return combo_label, corr_label, cum_threshold


def load_scenario3(scenario3_path, preprocessed_path, dataset_type):
    combo_label, corr_label, cum_threshold = find_best_combo_scenario3(scenario3_path)
    if combo_label is None:
        return None, None, None, None

    json_file = os.path.join(scenario3_path, combo_label, 'selected_features.json')
    if not os.path.exists(json_file):
        print(f"   ERROR: {json_file} not found!")
        return None, None, None, None

    with open(json_file) as f:
        info = json.load(f)

    selected = info['features']
    print(f"\n   Loaded {len(selected)} features from {combo_label}/selected_features.json")

    path    = os.path.join(preprocessed_path, f'dataset_{dataset_type}_transformed.csv')
    df_full = pd.read_csv(path)

    available = [feat for feat in selected if feat in df_full.columns]
    if len(available) < len(selected):
        print(f"   WARNING: {len(selected) - len(available)} features missing, using {len(available)}")

    X = df_full[available]
    y = df_full['label']

    print(f"   X shape  : {X.shape}")
    print(f"   Classes  : {y.unique().tolist()}")
    return X, y, combo_label, len(available)


# ====================================================================
# UMAP EMBEDDING
# ====================================================================

def create_umap_embedding(X, y, scenario_name):
    print(f"\n   Creating UMAP for {scenario_name}...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reducer  = umap.UMAP(**CONFIG['umap_settings'])
    emb      = reducer.fit_transform(X_scaled)

    umap_df  = pd.DataFrame(emb, columns=['UMAP_1', 'UMAP_2'])
    umap_df['Class']    = y.reset_index(drop=True)
    umap_df['Scenario'] = scenario_name

    print(f"      Embedding : {emb.shape}")
    print(f"      UMAP_1    : [{emb[:,0].min():.2f}, {emb[:,0].max():.2f}]")
    print(f"      UMAP_2    : [{emb[:,1].min():.2f}, {emb[:,1].max():.2f}]")

    return emb, umap_df, X_scaled


# ====================================================================
# HELPERS
# ====================================================================

def _out(filename):
    return os.path.join(CONFIG['folders']['output'], filename)


def _class_color(cls):
    return CONFIG['class_colors'].get(str(cls), '#7f7f7f')


def _add_class_traces(fig, umap_df, show_legend, row=1, col=1):
    """Adiciona um trace por classe com cores fixas."""
    ps      = CONFIG['plot_settings']
    classes = sorted(umap_df['Class'].unique())
    for cls in classes:
        data = umap_df[umap_df['Class'] == cls]
        fig.add_trace(
            go.Scatter(
                x=data['UMAP_1'], y=data['UMAP_2'],
                mode='markers', name=str(cls),
                marker=dict(
                    size=ps['point_size'],
                    color=_class_color(cls),
                    opacity=ps['opacity'],
                ),
                legendgroup=str(cls),
                showlegend=show_legend,
            ),
            row=row, col=col,
        )


# ====================================================================
# PLOTS
# ====================================================================

def plot_single_umap(umap_df, title, filename):
    ps  = CONFIG['plot_settings']

    fig = make_subplots(rows=1, cols=1)
    _add_class_traces(fig, umap_df, show_legend=True)

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=ps['title_size'])),
        width=ps['width'], height=ps['height'],
        font=dict(size=ps['font_size']),
        legend=dict(
            orientation='v', yanchor='top', y=1, xanchor='left', x=1.01,
            font=dict(size=ps['font_size']),
        ),
        xaxis=dict(
            title=dict(text='UMAP 1', font=dict(size=ps['font_size'])),
            tickfont=dict(size=ps['font_size']),
        ),
        yaxis=dict(
            title=dict(text='UMAP 2', font=dict(size=ps['font_size'])),
            tickfont=dict(size=ps['font_size']),
        ),
    )
    fig.write_html(_out(filename))
    print(f"      Saved: {filename}")


def plot_comparative_umaps(entries, dataset_type):
    """Plot comparativo 1×3 com títulos sem sobreposição."""
    print(f"\n Creating comparative plot...")
    ps = CONFIG['plot_settings']

    n_cols = len(entries)

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[e[1] for e in entries],
        horizontal_spacing=0.10,
    )

    for col, (umap_df, _) in enumerate(entries, 1):
        _add_class_traces(fig, umap_df, show_legend=(col == 1), row=1, col=col)
        fig.update_xaxes(
            title=dict(text='UMAP 1', font=dict(size=ps['font_size'])),
            tickfont=dict(size=ps['font_size']),
            row=1, col=col,
        )

    fig.update_yaxes(
        title=dict(text='UMAP 2', font=dict(size=ps['font_size'])),
        tickfont=dict(size=ps['font_size']),
        row=1, col=1,
    )

    # Dobrar fonte dos subplot_titles e empurrar para cima (evita sobreposição)
    for ann in fig.layout.annotations:
        ann.font = dict(size=ps['font_size'])
        ann.y   += 0.06

    fig.update_layout(
        height=800,
        width=2200,
        margin=dict(t=180),   # margem top para título global + subtítulos
        title=dict(
            text=f'UMAP Comparison — Scenarios 1, 2, 3 ({dataset_type.title()} set)',
            x=0.5,
            y=0.97,
            font=dict(size=ps['title_size']),
        ),
        font=dict(size=ps['font_size']),
        showlegend=True,
        legend=dict(
            orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.01,
            font=dict(size=ps['font_size']),
        ),
    )

    fname = f'umap_comparison_{dataset_type}.html'
    fig.write_html(_out(fname))
    print(f"   Saved: {fname}")


def plot_variance_analysis(scaled_datasets, dataset_type):
    print(f"\n Creating variance analysis...")
    ps   = CONFIG['plot_settings']
    rows = []
    for name, X_scaled, n_feat in scaled_datasets:
        n_comp = min(2, X_scaled.shape[1], X_scaled.shape[0] - 1)
        if n_comp < 1:
            rows.append({'name': name, 'variance': 0.0, 'n_features': n_feat})
            continue
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)
        var = float(np.sum(pca.explained_variance_ratio_[:2]))
        rows.append({'name': name, 'variance': var, 'n_features': n_feat})

    fig = go.Figure(go.Bar(
        x=[r['name'] for r in rows],
        y=[r['variance'] for r in rows],
        marker_color=['lightblue', 'lightgreen', 'lightcoral'],
        text=[f"{r['n_features']} features<br>{r['variance']:.1%}" for r in rows],
        textposition='auto',
        textfont=dict(size=ps['font_size']),
    ))
    fig.update_layout(
        title=dict(
            text=f'Explained Variance (First 2 PCA Components) — {dataset_type.title()} set',
            x=0.5, font=dict(size=ps['title_size']),
        ),
        xaxis=dict(
            title=dict(text='Scenario', font=dict(size=ps['font_size'])),
            tickfont=dict(size=ps['font_size']),
        ),
        yaxis=dict(
            title=dict(text='Explained Variance Ratio', font=dict(size=ps['font_size'])),
            tickfont=dict(size=ps['font_size']),
        ),
        height=500, width=900,
        font=dict(size=ps['font_size']),
    )
    fname = f'variance_analysis_{dataset_type}.html'
    fig.write_html(_out(fname))
    print(f"   Saved: {fname}")


# ====================================================================
# MAIN
# ====================================================================

def main():
    print(f"\n{'='*70}")
    print(" UMAP GENERATOR — SCENARIOS 1, 2, 3")
    print(f"{'='*70}")

    dt        = CONFIG['dataset_type']
    pre       = CONFIG['folders']['preprocessed']
    sc3       = CONFIG['folders']['scenario3']
    vars_list = CONFIG['sc1_variables']

    os.makedirs(CONFIG['folders']['output'], exist_ok=True)

    X1, y1 = load_scenario1(pre, dt, vars_list)
    if X1 is None:
        return

    X2, y2 = load_scenario2(pre, dt)
    if X2 is None:
        return

    print(f"\n{'='*70}")
    print(" LOADING SCENARIO 3: BEST COMBO FEATURES")
    print(f"{'='*70}")
    X3, y3, combo_label, n3 = load_scenario3(sc3, pre, dt)
    if X3 is None:
        return

    print(f"\n{'='*70}")
    print(" CREATING UMAP EMBEDDINGS")
    print(f"{'='*70}")

    emb1, umap_df1, X1_sc = create_umap_embedding(X1, y1, 'Scenario 1')
    emb2, umap_df2, X2_sc = create_umap_embedding(X2, y2, 'Scenario 2')
    emb3, umap_df3, X3_sc = create_umap_embedding(X3, y3, 'Scenario 3')

    print(f"\n{'='*70}")
    print(" GENERATING INDIVIDUAL PLOTS")
    print(f"{'='*70}")

    plot_single_umap(
        umap_df1,
        f'Scenario 1: Raw Sensor Data — flatten ({X1.shape[1]} features)',
        f'umap_scenario1_{dt}.html',
    )
    plot_single_umap(
        umap_df2,
        f'Scenario 2: All Transformed Features ({X2.shape[1]} features)',
        f'umap_scenario2_{dt}.html',
    )
    plot_single_umap(
        umap_df3,
        f'Scenario 3: ReliefF+ANOVA — {combo_label} ({n3} features)',
        f'umap_scenario3_{dt}.html',
    )

    print(f"\n{'='*70}")
    print(" GENERATING COMPARATIVE PLOT")
    print(f"{'='*70}")

    entries = [
        (umap_df1, f'Scenario 1: Raw flatten<br>({X1.shape[1]} features)'),
        (umap_df2, f'Scenario 2: All transformed<br>({X2.shape[1]} features)'),
        (umap_df3, f'Scenario 3: {combo_label}<br>({n3} features)'),
    ]
    plot_comparative_umaps(entries, dt)

    plot_variance_analysis(
        [('Scenario 1', X1_sc, X1.shape[1]),
         ('Scenario 2', X2_sc, X2.shape[1]),
         ('Scenario 3', X3_sc, n3)],
        dt,
    )

    print(f"\n{'='*70}")
    print(" COMPLETED")
    print(f"{'='*70}")
    print(f"   Scenario 1 : {X1.shape[1]} features (raw flatten — Fx/Fy/Fz/Tx/Ty/Tz timesteps)")
    print(f"   Scenario 2 : {X2.shape[1]} features (all transformed)")
    print(f"   Scenario 3 : {n3} features  [{combo_label}] — best F1_Macro, fewest features")
    print(f"\n   Output: {os.path.abspath(CONFIG['folders']['output'])}/")
    print(f"   Files:")
    print(f"     umap_scenario1_{dt}.html")
    print(f"     umap_scenario2_{dt}.html")
    print(f"     umap_scenario3_{dt}.html")
    print(f"     umap_comparison_{dt}.html")
    print(f"     variance_analysis_{dt}.html")


if __name__ == '__main__':
    main()