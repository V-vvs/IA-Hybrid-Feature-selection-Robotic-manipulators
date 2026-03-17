
import argparse, ast, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data',   type=str, default=None,
                    help='Folder containing dataset_train/test_transformed.csv')
parser.add_argument('--output', type=str, default=None,
                    help='Output folder for figures and predictions')
parser.add_argument('--results_dir', type=str, default=None,
                    help='Folder containing model_results_*.csv and selected_features_*.json files')
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent

def find_dir(name_pattern):
    """Busca recursiva por pasta ou arquivo."""
    for root in [Path.cwd(), SCRIPT_DIR]:
        r = root
        for _ in range(4):
            found = list(r.rglob(name_pattern))
            if found:
                return found[0].parent if found[0].is_file() else found[0]
            r = r.parent
    return None

# ── Locate folders ──────────────────────────────────────────────────────────────
data_dir    = Path(args.data).resolve()       if args.data       else find_dir('dataset_train_transformed.csv')
results_dir = Path(args.results_dir).resolve() if args.results_dir else find_dir('model_results_cor*.csv')
# Output folder:
def _find_output_dir():
    for base in [Path.cwd(), SCRIPT_DIR]:
        candidate = base.parent / 'ML_Results_Scenario3_GridSearch'
        if candidate.exists():
            return candidate
        candidate2 = base / 'ML_Results_Scenario3_GridSearch'
        if candidate2.exists():
            return candidate2
    # Fallback
    return SCRIPT_DIR.parent / 'ML_Results_Scenario3_GridSearch'

out_dir     = Path(args.output).resolve() if args.output else _find_output_dir()

if data_dir is None:
    raise FileNotFoundError("dataset_train_transformed.csv not found. Use --data.")
if results_dir is None:
    raise FileNotFoundError("model_results_*.csv not found. Use --results_dir.")

out_dir.mkdir(parents=True, exist_ok=True)
print(f"Data    : {data_dir}")
print(f"Results : {results_dir}")
print(f"Output  : {out_dir}")

# ── Per-model file mapping ───────────────────────────────────────────────────────
# Maps model
MODEL_FILES = {
    'KNN':      ('model_results_cor80_cum95_bestKNN.csv',   'selected_features_cor80_cum95_bestKNN.json'),
    'SVM':      ('model_results_cor90_cum80_bestSVM.csv',   'selected_features_cor90_cum80_bestSVM.json'),
    'MLP':      ('model_results_cor95_cum80_bestMLP.csv',   'selected_features_cor95_cum80_bestMLP.json'),
    'LightGBM': ('model_results_cor75_cum85_bestLGBM.csv',  'selected_features_cor75_cum85_bestLGBM.json'),
}

# ── Load datasets ───────────────────────────────────────────────────────────────
print("\nLoading datasets...")
train_df = pd.read_csv(data_dir / 'dataset_train_transformed.csv')
test_df  = pd.read_csv(data_dir / 'dataset_test_transformed.csv')
print(f"  Train: {train_df.shape}  |  Test: {test_df.shape}")

classes = ['collision', 'normal', 'obstruction']

# ── Helpers ─────────────────────────────────────────────────────────────────────
def parse_best_params(params_str):
    """Converte string do Best_Params para dict."""
    try:
        return ast.literal_eval(params_str)
    except Exception:
        return {}

# ── KNN weight functions ─────────────
def _knn_weight_inv_dist(distances):
    return 1.0 / (distances + 1e-6)

def _knn_weight_exp_dist(distances):
    return np.exp(-distances)

RANDOM_SEED = 42
MLP_ARCH_MAP = {
    '50':      (50,),
    '75':      (75,),
    '100':     (100,),
    '150':     (150,),
    '200':     (200,),
    '75_100':  (75, 100),
    '100_50':  (100, 50),
    '150_75':  (150, 75),
    '200_150': (200, 150),
}

def build_knn(params):
    wc     = params.get('weights', 'distance')
    metric = params.get('metric', 'minkowski')
    if wc == 'inv_dist':
        weights = _knn_weight_inv_dist
    elif wc == 'exp_dist':
        weights = _knn_weight_exp_dist
    else:
        weights = wc   
    algorithm = 'brute' if metric == 'cosine' else 'kd_tree'
    return KNeighborsClassifier(
        n_neighbors = params.get('n_neighbors', 5),
        metric      = metric,
        p           = int(params.get('p', 2)),
        weights     = weights,
        algorithm   = algorithm,
    )

def build_svm(params):
    sp = dict(
        kernel       = params.get('kernel', 'rbf'),
        C            = params.get('C', 1.0),
        tol          = 1.0,
        class_weight = params.get('class_weight', None),
        probability  = True,
        random_state = RANDOM_SEED,
    )
    if params.get('kernel') == 'rbf':
        sp['gamma'] = params['gamma']   
    return SVC(**sp)

def build_mlp(params):
    arch    = params.get('hidden_layer_sizes_key', '100')
    sizes   = MLP_ARCH_MAP.get(arch, (100,))
    solver  = params.get('solver', 'lbfgs')
    lr_sched = params.get('lr_schedule', 'constant')
    if 'invscale' in lr_sched:
        lr_sched = 'invscaling'
    mp = dict(
        hidden_layer_sizes  = sizes,
        activation          = params.get('activation', 'tanh'),
        solver              = solver,
        alpha               = params.get('alpha', 1e-4),
        max_iter            = params.get('max_iter', 1000),
        tol                 = 1e-4,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 20,
        random_state        = RANDOM_SEED,
    )
    if solver in ('adam', 'sgd'):
        mp['learning_rate'] = lr_sched
    return MLPClassifier(**mp)

def build_lgbm(params, feature_names=None):
    lgb_keys = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
                'min_child_samples', 'min_data_in_leaf',
                'feature_fraction', 'bagging_fraction', 'bagging_freq']
    lp = {k: params[k] for k in lgb_keys if k in params}
    # cast integer fields
    for k in ['n_estimators','num_leaves','max_depth','min_child_samples',
              'min_data_in_leaf','bagging_freq']:
        if k in lp:
            lp[k] = int(lp[k])
    lp.update(
        class_weight  = 'balanced',
        verbosity     = -1,
        n_jobs        = 1,
        force_col_wise= True,
        random_state  = RANDOM_SEED,
        deterministic = True,
        force_row_wise= False,
    )
    return lgb.LGBMClassifier(**lp)

# builders called directly per model

# ── Retrain and predict ──────────────────────────────────────────────────────────
results = {}   # model -> {'cm', 'acc', 'f1', 'corr_l', 'cum_t', 'n_feat'}

print()
for model in ['KNN', 'SVM', 'MLP', 'LightGBM']:
    csv_name, json_name = MODEL_FILES[model]

    # Locate files 
    csv_file  = results_dir / csv_name
    json_file = results_dir / json_name
    if not csv_file.exists():
        found = list(SCRIPT_DIR.rglob(csv_name)) + list(Path.cwd().rglob(csv_name))
        csv_file = found[0] if found else csv_file
    if not json_file.exists():
        found = list(SCRIPT_DIR.rglob(json_name)) + list(Path.cwd().rglob(json_name))
        json_file = found[0] if found else json_file

    if not csv_file.exists():
        print(f"[{model}] SKIP — {csv_name} not found")
        continue
    if not json_file.exists():
        print(f"[{model}] SKIP — {json_name} not found")
        continue

    print(f"[{model}] Loading configuration...")

    # Read hyperparameters from model row
    df_res = pd.read_csv(csv_file)
    row    = df_res[df_res['Model'] == model].iloc[0]
    params = parse_best_params(row['Best_Params'])
    corr_l = str(round(float(row['Corr_Threshold']), 2))
    cum_t  = float(row['Cum_Threshold'])
    n_feat = int(row['Num_Features'])
    f1_val = float(row['F1_Macro'])
    acc_val= float(row['Accuracy'])

    print(f"  rho={corr_l}  tau_cum={cum_t}  n_feat={n_feat}")
    print(f"  Params: {params}")

    # Load selected features
    with open(json_file) as f:
        feat_info = json.load(f)
    features = feat_info['features']

    # Filter to available features
    available = [ft for ft in features if ft in train_df.columns]
    if len(available) < len(features):
        print(f"  WARNING: {len(features)-len(available)} features missing from dataset")

    X_train = train_df[available].values
    y_train = train_df['label'].values
    X_test  = test_df[available].values
    y_test  = test_df['label'].values

    # Normalize (StandardScaler fit on training set)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train)
    X_te_sc  = scaler.transform(X_test)

    # Build and train
    print(f"  Training {model}...")
    acc_expected = float(row['Accuracy'])

    if model == 'KNN':
        clf = build_knn(params)
        clf.fit(X_tr_sc, y_train)
        y_pred = clf.predict(X_te_sc)
    elif model == 'SVM':
        clf = build_svm(params)
        clf.fit(X_tr_sc, y_train)
        y_pred = clf.predict(X_te_sc)
    elif model == 'MLP':
        clf = build_mlp(params)
        clf.fit(X_tr_sc, y_train)
        y_pred = clf.predict(X_te_sc)
    elif model == 'LightGBM':
        clf = build_lgbm(params)
        Xtr_df = pd.DataFrame(X_tr_sc, columns=available)
        Xte_df = pd.DataFrame(X_te_sc, columns=available)
        clf.fit(Xtr_df, y_train)
        y_pred = clf.predict(Xte_df)

    acc_real = np.mean(y_pred == y_test)
    print(f"  Acc expected : {acc_expected:.4f}")
    print(f"  Acc obtained : {acc_real:.4f}")

    # Save predictions
    pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    pred_path = out_dir / f'predictions_{model}.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f"  Acc (test) = {acc_real:.4f}  |  Saved: {pred_path.name}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    results[model] = {
        'cm':     cm,
        'acc':    acc_real,
        'f1':     f1_val,
        'corr_l': corr_l,
        'cum_t':  cum_t,
        'n_feat': n_feat,
    }

# ── Figure ──────────────────────────────────────────────────────────────────────
print("\nGenerating confusion matrix figure...")

plt.rcParams.update({
    'font.family':  'serif',
    'font.serif':   ['DejaVu Serif', 'Times New Roman', 'Times'],
    'text.usetex':  False,
})

cmap_cm = LinearSegmentedColormap.from_list(
    'cm_blue', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#084594'])

models_order = ['KNN', 'SVM', 'MLP', 'LightGBM']

fig, axes = plt.subplots(
    2, 2,
    figsize=(24, 22),
    gridspec_kw={
        'hspace': 0.65,
        'wspace': 0.38,
        'left':   0.08,
        'right':  0.96,
        'top':    0.87,
        'bottom': 0.08,
    }
)
fig.suptitle('Confusion Matrices — Best Configuration per Model (Test Set)',
             fontsize=30, fontweight='bold', y=0.97)

axes_flat = axes.flatten()

for idx, model in enumerate(models_order):
    ax = axes_flat[idx]

    if model not in results:
        ax.set_visible(False)
        continue

    r   = results[model]
    cm  = r['cm']

    im = ax.imshow(cm, cmap=cmap_cm, aspect='auto', interpolation='nearest')

    vmax_cm = cm.max() if cm.max() > 0 else 1
    for ri in range(len(classes)):
        for ci in range(len(classes)):
            val = cm[ri, ci]
            brightness = val / vmax_cm
            color = 'white' if brightness > 0.55 else '#1a1a1a'
            ax.text(ci, ri, str(int(val)),
                    ha='center', va='center',
                    fontsize=30, fontweight='bold', color=color)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=22)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=22)
    ax.set_xlabel('Predicted', fontsize=24, labelpad=8)
    ax.set_ylabel('Actual',    fontsize=24, labelpad=8)

    ax.set_title(
        f'{model}\n'
        f'Acc={r["acc"]:.3f}  F1={r["f1"]:.3f}\n'
        f'\u03c1={r["corr_l"]}  \u03c4_cum={r["cum_t"]:.2f}  ({r["n_feat"]} feat.)',
        fontsize=24, fontweight='bold', pad=10
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=20)

    # Minor grid
    ax.set_xticks(np.arange(len(classes)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

out_fig = out_dir / 'fig_confusion_best.png'
plt.savefig(out_fig, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()
print(f"\nSaved: {out_fig}")
print("\nDone.")
