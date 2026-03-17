import os
import sys

# ====================================================================
# REPRODUCIBILITY: env vars MUST be set before ANY library import.
# ====================================================================
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED']   = str(RANDOM_SEED)
os.environ['OMP_NUM_THREADS']  = '1'   
os.environ['PYTHONUNBUFFERED'] = '1'
# ====================================================================

import pandas as pd
import numpy as np
import random
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score)
from scipy.stats import chi2

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library available")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP library not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# ====================================================================
# KNN WEIGHT FUNCTIONS
# ====================================================================

def _knn_weight_inv_dist(distances):
    return 1.0 / (distances + 1e-6)

def _knn_weight_exp_dist(distances):
    return np.exp(-distances)

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f"Random seeds set to {RANDOM_SEED} for complete reproducibility")
print(f"OMP_NUM_THREADS=1 set before lightgbm import (determinismo LightGBM)")

optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.ioff()


# ====================================================================
# MLP ARCHITECTURE MAP — defined ONCE, used in objective and rebuild.
# ====================================================================
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

script_version = "Scenario1_RawData_Reproducible_AlignedHyperspace"
print(f"Starting ML Pipeline {script_version}")
print("SCENARIO 1: Raw Sensor Data as Features — No Feature Selection")
print("Hyperparameters optimized with OPTUNA (aligned hyperspace with Scenario 3/4)")
print("ENHANCED: McNemar Test + Bootstrap Confidence Intervals + SHAP")

# ====================================================================
# CONFIGURATION
# ====================================================================
SKIP_LIGHTGBM       = False
APPLY_SHAP_ANALYSIS = True
SHAP_SAMPLE_SIZE    = 100

# Aligned with Scenario 3/4: 100 trials each, 10-fold CV
N_TRIALS_MAP = {'KNN': 50, 'SVM': 50, 'MLP': 50, 'LightGBM': 50} 
CV_FOLDS     = 10  

print(f"Configuration: SKIP_LIGHTGBM={SKIP_LIGHTGBM}, CV_FOLDS={CV_FOLDS}")
print(f"Optuna Trials: {N_TRIALS_MAP}")
print(f"SHAP Analysis: {APPLY_SHAP_ANALYSIS and SHAP_AVAILABLE}")

output_folder      = f'ML_Results_Scenario1_RawData_{script_version}'
plots_folder       = os.path.join(output_folder, 'Plots')
shap_folder        = os.path.join(output_folder, 'SHAP_Analysis')
hyperparams_folder = os.path.join(output_folder, 'Best_Hyperparameters')

for folder in [output_folder, plots_folder, shap_folder, hyperparams_folder]:
    os.makedirs(folder, exist_ok=True)

# ====================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ====================================================================

def optimize_with_optuna(X_train, y_train, model_name, cv_folds=10):
    """
    Optimize hyperparameters using Optuna — reprodutivel por modelo.
    Hyperspace aligned with Scenario 3/4 for cross-scenario comparability.
    """
    print(f"  Optimizing {model_name} with Optuna...", end=" ")

    seed     = RANDOM_SEED   # seed único para todos os modelos (alinhado com Sc3)
    cv       = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    n_trials = N_TRIALS_MAP[model_name]

    X_input = (pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame)
               else X_train)

    def objective(trial):

        # ── KNN ──────────────────────────────────────────────────────────
        if model_name == 'KNN':
            n_neighbors   = trial.suggest_categorical('n_neighbors', [1, 2, 3, 5, 7, 9])
            metric        = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
            p_val         = trial.suggest_categorical('p', [1, 1.2, 1.5, 2, 2.5, 3])
            weight_choice = trial.suggest_categorical(
                'weights', ['distance', 'inv_dist', 'exp_dist'])

            weights   = (_knn_weight_inv_dist if weight_choice == 'inv_dist'
                         else (_knn_weight_exp_dist if weight_choice == 'exp_dist'
                               else weight_choice))
            algorithm = 'brute' if metric == 'cosine' else 'kd_tree'
            model     = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights=weights,
                metric=metric, p=p_val, algorithm=algorithm)

        # ── SVM ──────────────────────────────────────────────────────────
        elif model_name == 'SVM':
            kernel = trial.suggest_categorical('kernel', ['rbf'])
            params = dict(C=trial.suggest_float('C', 0.1, 15, log=True),
                          kernel=kernel,
                          class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
                          probability=True,
                          random_state=seed)
            if kernel == 'rbf':
                gc = trial.suggest_categorical('gamma_choice', ['log_val'])
                params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
            model = SVC(**params)

        # ── MLP ──────────────────────────────────────────────────────────
        elif model_name == 'MLP':
            arch      = trial.suggest_categorical('hidden_layer_sizes_key', list(MLP_ARCH_MAP))
            solver    = trial.suggest_categorical('solver', ['lbfgs'])
            _lr_sched = trial.suggest_categorical('lr_schedule', ['invscale'])
            params = dict(
                hidden_layer_sizes=MLP_ARCH_MAP[arch],
                activation=trial.suggest_categorical('activation', ['tanh', 'relu']),
                solver=solver,
                alpha=trial.suggest_float('alpha', 1e-3, 1.0, log=True),
                max_iter=trial.suggest_categorical('max_iter', [1000, 500]),
                tol=1e-4, early_stopping=True,
                validation_fraction=0.2, n_iter_no_change=20,
                random_state=seed)
            model = MLPClassifier(**params)

        # ── LightGBM ─────────────────────────────────────────────────────
        elif model_name == 'LightGBM':
            params = dict(
                n_estimators     =trial.suggest_int('n_estimators',       1050,  1100),
                learning_rate    =trial.suggest_float('learning_rate',    0.05,  0.6,  log=True),
                num_leaves       =trial.suggest_int('num_leaves',         9,     10),
                max_depth        =trial.suggest_int('max_depth',          5,     7),
                subsample        =trial.suggest_float('subsample',        0.5,   0.7),
                colsample_bytree =trial.suggest_float('colsample_bytree', 0.9,   1.0),
                reg_alpha        =trial.suggest_float('reg_alpha',        0.05,  0.9,  log=True),
                reg_lambda       =trial.suggest_float('reg_lambda',       0.3,   1.0,  log=True),
                min_child_samples=trial.suggest_int('min_child_samples',  30,    50),
                min_data_in_leaf =trial.suggest_int('min_data_in_leaf',   15,    30),
                feature_fraction =trial.suggest_float('feature_fraction', 0.5,   1.0),
                bagging_fraction =trial.suggest_float('bagging_fraction', 0.6,   1.0),
                bagging_freq     =trial.suggest_int('bagging_freq',       6,     10),
                class_weight='balanced',
                verbosity=-1,
                n_jobs=1,
                force_col_wise=True,
                random_state=seed,
                deterministic=True,
                force_row_wise=False)
            model = lgb.LGBMClassifier(**params)

        try:
            scores = cross_val_score(model, X_input, y_train, cv=cv,
                                     scoring='f1_macro', n_jobs=1)
            trial.set_user_attr('cv_max', float(scores.max()))
            trial.set_user_attr('cv_std', float(scores.std()))
            return float(scores.mean())
        except Exception:
            return 0.0

    sampler = TPESampler(seed=seed)
    study   = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)

    best_params  = study.best_params
    best_trial   = study.best_trial
    _cv_mean     = study.best_value
    _cv_std      = best_trial.user_attrs.get('cv_std',  float('nan'))
    _cv_max_fold = best_trial.user_attrs.get('cv_max',  float('nan'))

    print(f"CV_mean={_cv_mean:.4f}  best_fold={_cv_max_fold:.4f}")

    # ── Rebuild best model ────────────────────────────────────────────────────
    if model_name == 'KNN':
        wc = best_params['weights']
        fw = (_knn_weight_inv_dist if wc == 'inv_dist'
              else (_knn_weight_exp_dist if wc == 'exp_dist' else wc))
        mt = best_params['metric']
        best_model = KNeighborsClassifier(
            n_neighbors=best_params['n_neighbors'], weights=fw,
            metric=mt, p=best_params['p'],
            algorithm='brute' if mt == 'cosine' else 'kd_tree')

    elif model_name == 'SVM':
        sp = dict(C=best_params['C'], kernel=best_params['kernel'],
                  tol=1.0, class_weight=best_params['class_weight'], 
                  probability=True, random_state=seed)
        if best_params['kernel'] == 'rbf':
            sp['gamma'] = best_params['gamma']
        best_model = SVC(**sp)

    elif model_name == 'MLP':
        solver = best_params['solver']
        mp = dict(
            hidden_layer_sizes=MLP_ARCH_MAP[best_params['hidden_layer_sizes_key']],
            activation=best_params['activation'], solver=solver,
            alpha=best_params['alpha'], max_iter=best_params['max_iter'],
            tol=1e-4, early_stopping=True,
            validation_fraction=0.2, n_iter_no_change=20,
            random_state=seed)
        if solver in ('adam', 'sgd'):
            mp['learning_rate_init'] = best_params.get('learning_rate_init', 1e-3)
            mp['batch_size']         = best_params.get('batch_size', 64)
        if solver == 'adam':
            lr_sched = best_params.get('lr_schedule', 'constant')
            mp['learning_rate'] = lr_sched if lr_sched != 'invscaling' else 'constant'
            mp['beta_1']        = best_params.get('beta_1', 0.9)
            mp['beta_2']        = best_params.get('beta_2', 0.999)
        elif solver == 'sgd':
            mp['learning_rate'] = best_params.get('lr_schedule', 'constant')
            mp['momentum']      = best_params.get('momentum', 0.9)
        best_model = MLPClassifier(**mp)

    elif model_name == 'LightGBM':
        lgb_keys = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                    'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
                    'min_child_samples', 'min_data_in_leaf',
                    'feature_fraction', 'bagging_fraction', 'bagging_freq']
        lp = {k: best_params[k] for k in lgb_keys if k in best_params}
        lp.update(class_weight='balanced', verbosity=-1, n_jobs=1,
                  force_col_wise=True, random_state=seed,
                  deterministic=True, force_row_wise=False)
        best_model = lgb.LGBMClassifier(**lp)

    # ── Recompute CV on rebuilt model ─────────────────────────────────────────
    try:
        cv_rebuilt = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores_rebuilt = cross_val_score(best_model, X_input, y_train,
                                            cv=cv_rebuilt, scoring='f1_macro', n_jobs=1)
        _cv_mean     = float(cv_scores_rebuilt.mean())
        _cv_std      = float(cv_scores_rebuilt.std())
        _cv_max_fold = float(cv_scores_rebuilt.max())
        print(f"  {model_name} CV (rebuilt): mean={_cv_mean:.4f} +/- {_cv_std:.4f}  "
              f"best_fold={_cv_max_fold:.4f}")
    except Exception:
        pass

    return best_model, best_params, _cv_mean, _cv_std, _cv_max_fold

# ====================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ====================================================================

def calculate_confidence_intervals(y_true, y_pred, n_bootstrap=1000,
                                   confidence=0.95, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)

    bootstrap_metrics = {
        'accuracy': [], 'f1_macro': [], 'f1_weighted': [],
        'precision_macro': [], 'recall_macro': [],
        'precision_weighted': [], 'recall_weighted': []
    }

    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices     = rng.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        bootstrap_metrics['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
        bootstrap_metrics['f1_macro'].append(f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        bootstrap_metrics['f1_weighted'].append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        bootstrap_metrics['precision_macro'].append(precision_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        bootstrap_metrics['recall_macro'].append(recall_score(y_true_boot, y_pred_boot, average='macro', zero_division=0))
        bootstrap_metrics['precision_weighted'].append(precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
        bootstrap_metrics['recall_weighted'].append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))

    alpha = 1 - confidence
    return {
        k: {
            'lower': np.percentile(v, (alpha / 2) * 100),
            'upper': np.percentile(v, (1 - alpha / 2) * 100)
        }
        for k, v in bootstrap_metrics.items()
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b):
    y_true   = np.array(y_true)
    y_pred_a = np.array(y_pred_a)
    y_pred_b = np.array(y_pred_b)

    ca  = (y_pred_a == y_true)
    cb  = (y_pred_b == y_true)
    n01 = int(np.sum( ca & ~cb))
    n10 = int(np.sum(~ca &  cb))
    n00 = int(np.sum(~ca & ~cb))
    n11 = int(np.sum( ca &  cb))

    if (n01 + n10) == 0:
        stat, pval = 0.0, 1.0
    else:
        stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        pval = 1 - chi2.cdf(stat, df=1)

    return {
        'model_a': name_a, 'model_b': name_b,
        'n01': n01, 'n10': n10, 'n00': n00, 'n11': n11,
        'statistic': float(stat), 'p_value': float(pval),
        'significant_at_0.05': pval < 0.05,
        'significant_at_0.01': pval < 0.01
    }


def perform_pairwise_mcnemar(predictions_dict, y_true, class_labels):
    print(f"\n{'='*70}")
    print("MCNEMAR'S TEST — PAIRWISE MODEL COMPARISON")
    print(f"{'='*70}")

    model_names = list(predictions_dict.keys())
    results     = []

    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            if predictions_dict[ma] is None or predictions_dict[mb] is None:
                print(f"  Skipping {ma} vs {mb}: one or both models have no predictions")
                continue

            try:
                res = mcnemar_test(y_true,
                                   predictions_dict[ma]['y_pred'],
                                   predictions_dict[mb]['y_pred'],
                                   ma, mb)
                results.append(res)

                sig = "**" if res['significant_at_0.01'] else ("*" if res['significant_at_0.05'] else "")
                print(f"\n   {ma} vs {mb}:")
                print(f"     Statistic: {res['statistic']:.4f}")
                print(f"     p-value:   {res['p_value']:.4f} {sig}")
                print(f"     Contingency: [{res['n00']}, {res['n01']}]")
                print(f"                  [{res['n10']}, {res['n11']}]")

                if res['significant_at_0.05']:
                    better = ma if res['n01'] > res['n10'] else mb
                    worse  = mb if res['n01'] > res['n10'] else ma
                    print(f"     -> {better} significantly BETTER than {worse}")
                else:
                    print(f"     -> No significant difference")

            except Exception as e:
                print(f"  Error comparing {ma} vs {mb}: {e}")

    print(f"\n   * p < 0.05, ** p < 0.01")
    return pd.DataFrame(results) if results else pd.DataFrame()

# ====================================================================
# DATA LOADING AND PREPROCESSING
# ====================================================================

DATASET_BASE_PATH  = './Preprocessed_Data'
DATASET_TRAIN_PATH = os.path.join(DATASET_BASE_PATH, 'dataset_train_original.csv')
DATASET_TEST_PATH  = os.path.join(DATASET_BASE_PATH, 'dataset_test_original.csv')

try:
    df_train_raw = pd.read_csv(DATASET_TRAIN_PATH)
    print(f"\nLoaded training dataset: {DATASET_TRAIN_PATH} — shape {df_train_raw.shape}")
    print(f"  NaN values: {df_train_raw.isna().sum().sum()}")
    print("  Class distribution in training data:")
    print(df_train_raw['label'].value_counts())
except Exception as e:
    print(f"Error loading training dataset from {DATASET_TRAIN_PATH}: {e}")
    sys.exit(1)

try:
    df_test_raw = pd.read_csv(DATASET_TEST_PATH)
    print(f"\nLoaded test dataset: {DATASET_TEST_PATH} — shape {df_test_raw.shape}")
    print(f"  NaN values: {df_test_raw.isna().sum().sum()}")
    print("  Class distribution in test data:")
    print(df_test_raw['label'].value_counts())
except Exception as e:
    print(f"Error loading test dataset from {DATASET_TEST_PATH}: {e}")
    sys.exit(1)

variables = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

# Parse string lists
for var in variables:
    df_train_raw[var] = df_train_raw[var].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_test_raw[var]  = df_test_raw[var].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

print(f"\nNaN values after parsing — TRAIN: {df_train_raw[variables].isna().sum().sum()}")
print(f"NaN values after parsing — TEST:  {df_test_raw[variables].isna().sum().sum()}")

# ====================================================================
# FLATTEN RAW DATA
# ====================================================================

def flatten_raw_data(df, vars_list, max_len=None, is_training=True):
    if is_training:
        current_max_len = 0
        for var in vars_list:
            valid = df[var].dropna()
            if not valid.empty:
                current_max_len = max(current_max_len, int(valid.apply(len).max()))
        if current_max_len == 0:
            print("All feature lists are empty or NaN in training data. Cannot flatten.")
            return pd.DataFrame(), 0
        max_len_to_use = current_max_len
        print(f"Flattening training data — max sequence length: {max_len_to_use} per variable.")
    else:
        if not max_len:
            print("max_len not provided for test data. Cannot flatten consistently.")
            return pd.DataFrame(), 0
        max_len_to_use = max_len
        print(f"Flattening test data using training max length: {max_len_to_use} per variable.")

    flattened = []
    for _, row in df.iterrows():
        row_features = {}
        for var in vars_list:
            values = np.array(row[var], dtype=float)
            if values.size == 0 or np.all(np.isnan(values)):
                padded = np.full(max_len_to_use, np.nan)
            else:
                padded = np.pad(values, (0, max_len_to_use - values.size),
                                'constant', constant_values=np.nan)
            for i, val in enumerate(padded):
                row_features[f'{var}_{i+1}'] = val
        flattened.append(row_features)

    df_flat = pd.DataFrame(flattened, index=df.index)
    df_flat = df_flat.replace([np.inf, -np.inf], np.nan)

    if df_flat.isnull().values.any():
        medians = df_flat.median(numeric_only=True).fillna(0)
        print(f"NaNs found — filling with column medians...")
        df_flat = df_flat.fillna(medians)

    if df_flat.isnull().values.any():
        print("Remaining NaNs after median fill — filling with 0.")
        df_flat = df_flat.fillna(0)

    return df_flat, max_len_to_use

# ====================================================================
# SHAP ANALYSIS FUNCTIONS
# ====================================================================

def create_variable_mapping(feature_names, vars_list):
    mapping = {}
    for feat in feature_names:
        for var in vars_list:
            if feat.startswith(f'{var}_'):
                mapping[feat] = var
                break
    return mapping


def aggregate_shap_by_variable(shap_values, feature_names, vars_list):
    mapping = create_variable_mapping(feature_names, vars_list)

    if shap_values.ndim == 3:
        n_s, n_f, n_c = shap_values.shape
        agg = np.zeros((n_s, len(vars_list), n_c))
        for i, var in enumerate(vars_list):
            idx = [j for j, f in enumerate(feature_names) if mapping.get(f) == var]
            if idx:
                agg[:, i, :] = np.sum(shap_values[:, idx, :], axis=1)
    else:
        n_s, n_f = shap_values.shape
        agg = np.zeros((n_s, len(vars_list)))
        for i, var in enumerate(vars_list):
            idx = [j for j, f in enumerate(feature_names) if mapping.get(f) == var]
            if idx:
                agg[:, i] = np.sum(shap_values[:, idx], axis=1)

    return agg


def calculate_shap_importance(model, X_sample, model_name, feature_names, vars_list,
                              class_names=None):
    print(f"  Calculating SHAP values for {model_name}...")
    try:
        if model_name == 'LightGBM':
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_sample)
        else:
            background = shap.utils.sample(X_sample, min(50, len(X_sample)))
            explainer  = shap.Explainer(model.predict, background)
            shap_obj   = explainer(X_sample)
            shap_vals  = shap_obj.values if hasattr(shap_obj, 'values') else shap_obj

        if isinstance(shap_vals, list):
            shap_vals = np.stack(shap_vals, axis=-1)

        agg = aggregate_shap_by_variable(shap_vals, feature_names, vars_list)

        # ── Overall importance (mean across classes) ──────────────────────────
        mean_abs = (np.mean(np.abs(agg), axis=(0, 2))
                    if agg.ndim == 3
                    else np.mean(np.abs(agg), axis=0))

        importance = dict(zip(vars_list, mean_abs))
        ranking    = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # ── Per-class importance (mean |SHAP| per variable per class) ─────────
        per_class_importance = {}
        if class_names is not None and agg.ndim == 3:
            for ci, cn in enumerate(class_names):
                if ci >= agg.shape[2]: continue
                ma = np.mean(np.abs(agg[:, :, ci]), axis=0)
                per_class_importance[cn] = dict(zip(vars_list, ma))
        elif class_names is not None and agg.ndim == 2:
            # binary — single output; assign to both classes with opposite sign convention
            ma = np.mean(np.abs(agg), axis=0)
            for cn in class_names:
                per_class_importance[cn] = dict(zip(vars_list, ma))

        print(f"    SHAP analysis completed for {model_name}")
        return agg, ranking, per_class_importance

    except Exception as e:
        print(f"    Error calculating SHAP for {model_name}: {e}")
        return None, None, {}


def plot_shap_importance(variable_ranking, model_name, scenario_name,
                         per_class_importance=None, class_names=None):
    if not variable_ranking:
        return

    # ── Overall bar plot (mean across classes) ────────────────────────────────
    vars_ranked, imp_vals = zip(*variable_ranking)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(vars_ranked, imp_vals, color=colors[:len(vars_ranked)])
    plt.title(f'SHAP Variable Importance — {model_name}\n{scenario_name}\n(mean across all classes)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Mean |SHAP Value|', fontsize=12)
    plt.xticks(rotation=45)

    for bar, val in zip(bars, imp_vals):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(imp_vals) * 0.01,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    ranking_text = "Ranking:\n" + "\n".join(
        [f"{i+1}. {v} ({s:.4f})" for i, (v, s) in enumerate(variable_ranking)])
    plt.text(0.02, 0.98, ranking_text, transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             fontsize=9)

    plt.tight_layout()
    fname = f'SHAP_Importance_{model_name}_{scenario_name.replace(" ", "_").replace("/", "_")}.png'
    try:
        plt.savefig(os.path.join(shap_folder, fname), dpi=180, bbox_inches='tight')
        print(f"  SHAP importance plot saved: {fname}")
    except Exception as e:
        print(f"  Error saving SHAP plot: {e}")
    plt.close()

    # ── Grouped bar plot: variables × class ───────────────────────────────────
    if per_class_importance and class_names:
        class_colors = ['#4C72B0', '#DD8452', '#55A868']
        n_var  = len(vars_ranked)
        n_cls  = len(class_names)
        bw     = 0.8 / n_cls
        x      = np.arange(n_var)

        fig, ax = plt.subplots(figsize=(max(10, n_var * 1.2), 6))
        for ci, cn in enumerate(class_names):
            if cn not in per_class_importance: continue
            vals   = [per_class_importance[cn].get(v, 0.0) for v in vars_ranked]
            offset = (ci - n_cls / 2 + 0.5) * bw
            ax.bar(x + offset, vals, bw,
                   label=cn, color=class_colors[ci % len(class_colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(vars_ranked, fontsize=11)
        ax.set_xlabel('Sensor Variable', fontsize=12)
        ax.set_ylabel('Mean |SHAP Value|', fontsize=12)
        ax.set_title(
            f'SHAP Variable Importance per Class — {model_name}\n{scenario_name}',
            fontsize=13, fontweight='bold')
        ax.legend(title='Class', fontsize=10, title_fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        fname2 = f'SHAP_Importance_PerClass_{model_name}_{scenario_name.replace(" ", "_").replace("/", "_")}.png'
        try:
            plt.savefig(os.path.join(shap_folder, fname2), dpi=180, bbox_inches='tight')
            print(f"  SHAP per-class plot saved: {fname2}")
        except Exception as e:
            print(f"  Error saving SHAP per-class plot: {e}")
        plt.close()


def plot_combined_shap_ranking(all_rankings, scenario_name):
    if not all_rankings:
        return
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'SHAP Variable Importance Comparison — {scenario_name}',
                 fontsize=16, fontweight='bold')
    axes  = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

    for idx, (mn, ranking) in enumerate(all_rankings.items()):
        if idx >= 4 or not ranking:
            continue
        vars_ranked, imp_vals = zip(*ranking)
        bars = axes[idx].bar(vars_ranked, imp_vals, color=colors[:len(vars_ranked)])
        axes[idx].set_title(mn, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Variables')
        axes[idx].set_ylabel('Mean |SHAP Value|')
        axes[idx].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, imp_vals):
            axes[idx].text(bar.get_x() + bar.get_width() / 2,
                           bar.get_height() + max(imp_vals) * 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    for idx in range(len(all_rankings), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fname = f'SHAP_Combined_{scenario_name.replace(" ", "_").replace("/", "_")}.png'
    try:
        plt.savefig(os.path.join(shap_folder, fname), dpi=180, bbox_inches='tight')
        print(f"  Combined SHAP rankings saved: {fname}")
    except Exception as e:
        print(f"  Error saving combined SHAP plot: {e}")
    plt.close()


def save_shap_rankings(all_rankings, scenario_name, vars_list,
                       all_per_class=None, class_names=None):
    if not all_rankings:
        return
    rows = []
    for mn, ranking in all_rankings.items():
        if ranking:
            pci = (all_per_class or {}).get(mn, {})
            for rank, (var, imp) in enumerate(ranking, 1):
                row = {'Model': mn, 'Variable': var,
                       'Rank': rank, 'SHAP_Importance_Overall': imp,
                       'Scenario': scenario_name}
                for cn in (class_names or []):
                    row[f'SHAP_{cn}'] = pci.get(cn, {}).get(var, float('nan'))
                rows.append(row)
    if not rows:
        return

    df_rank = pd.DataFrame(rows)
    fname   = f'SHAP_Rankings_{scenario_name.replace(" ", "_").replace("/", "_")}.csv'
    df_rank.to_csv(os.path.join(shap_folder, fname), index=False)

    summary = []
    for var in vars_list:
        sub = df_rank[df_rank['Variable'] == var]
        summary.append({'Variable':            var,
                        'Avg_Rank':            sub['Rank'].mean(),
                        'Avg_SHAP_Importance': sub['SHAP_Importance_Overall'].mean(),
                        'Std_Rank':            sub['Rank'].std(),
                        'Std_SHAP_Importance': sub['SHAP_Importance_Overall'].std()})

    df_sum = pd.DataFrame(summary).sort_values('Avg_Rank')
    sfname = f'SHAP_Summary_{scenario_name.replace(" ", "_").replace("/", "_")}.csv'
    df_sum.to_csv(os.path.join(shap_folder, sfname), index=False)

    print(f"  SHAP rankings saved: {fname}")
    print(f"  SHAP summary saved:  {sfname}")
    print("\n  SHAP VARIABLE IMPORTANCE SUMMARY:")
    print("  " + "="*50)
    for _, row in df_sum.iterrows():
        print(f"  {row['Variable']:4s} | Avg Rank: {row['Avg_Rank']:.1f} | "
              f"Avg Importance: {row['Avg_SHAP_Importance']:.4f}")
    print("  " + "="*50)


def perform_shap_analysis(trained_models, X_test_processed, feature_names, vars_list, scenario_name):
    if not SHAP_AVAILABLE:
        print("  SHAP not available, skipping analysis")
        return

    print(f"\n  Starting SHAP Analysis for {scenario_name}...")

    rng_shap    = np.random.RandomState(RANDOM_SEED + 9999)
    sample_size = min(SHAP_SAMPLE_SIZE, len(X_test_processed))
    idx_sample  = rng_shap.choice(len(X_test_processed), sample_size, replace=False)
    X_sample    = (X_test_processed.iloc[idx_sample]
                   if isinstance(X_test_processed, pd.DataFrame)
                   else X_test_processed[idx_sample])

    print(f"  Using {sample_size} samples for SHAP analysis (seed={RANDOM_SEED + 9999})")

    all_rankings   = {}
    all_per_class  = {}
    for mn, assets in trained_models.items():
        if assets and 'model' in assets:
            _, ranking, per_class = calculate_shap_importance(
                assets['model'], X_sample, mn, feature_names, vars_list,
                class_names=assets.get('class_names'))
            if ranking:
                all_rankings[mn]  = ranking
                all_per_class[mn] = per_class
                plot_shap_importance(ranking, mn, scenario_name,
                                     per_class_importance=per_class,
                                     class_names=assets.get('class_names'))

    if all_rankings:
        plot_combined_shap_ranking(all_rankings, scenario_name)
        save_shap_rankings(all_rankings, scenario_name, vars_list,
                           all_per_class=all_per_class,
                           class_names=next(iter(all_per_class.values()), {}).keys() if all_per_class else None)

    print(f"SHAP Analysis completed for {scenario_name}")

# ====================================================================
# FLATTEN DATA
# ====================================================================

print("\nFlattening raw sensor data for training and testing...")

X_train_flat, seq_len = flatten_raw_data(df_train_raw, variables, is_training=True)
if X_train_flat.empty:
    print("Flattened training DataFrame is empty. Exiting.")
    sys.exit(1)

X_test_flat, _ = flatten_raw_data(df_test_raw, variables, max_len=seq_len, is_training=False)
if X_test_flat.empty:
    print("Flattened test DataFrame is empty. Exiting.")
    sys.exit(1)

print(f"Flattened training shape : {X_train_flat.shape}")
print(f"Flattened test shape     : {X_test_flat.shape}")
print(f"Total raw features       : {X_train_flat.shape[1]}")

# ====================================================================
# TARGET PREPARATION
# ====================================================================

y_train_raw = df_train_raw['label']
y_test_raw  = df_test_raw['label']

le = LabelEncoder()
le.fit(pd.concat([y_train_raw, y_test_raw]).unique())

y_train_enc = le.transform(y_train_raw)
y_test_enc  = le.transform(y_test_raw)
class_names = list(le.classes_)
num_classes = len(class_names)

print(f"\nTarget: {num_classes} classes: {class_names}")
print(f"  Train class distribution: {dict(zip(class_names, np.bincount(y_train_enc)))}")
print(f"  Test  class distribution: {dict(zip(class_names, np.bincount(y_test_enc)))}")

# ====================================================================
# MODEL TRAINING WITH OPTUNA
# ====================================================================

def train_evaluate_models(X_tr, y_tr, X_te, y_te, scenario_name, class_labels):
    print(f"\nTraining/Evaluating: {scenario_name}")
    if X_tr.empty or X_te.empty:
        print(f"  Empty data for {scenario_name}. Skipping.")
        return {}, {}, {}

    feature_names = X_tr.columns.tolist()

    print(f"  Standardizing data...")
    scaler  = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=feature_names, index=X_tr.index)
    X_te_sc = pd.DataFrame(scaler.transform(X_te),     columns=feature_names, index=X_te.index)
    print(f"  Standardization completed.  Train: {X_tr_sc.shape}  Test: {X_te_sc.shape}")

    model_names = ['KNN', 'SVM', 'MLP']
    if not SKIP_LIGHTGBM:
        model_names.append('LightGBM')

    results_dict = {}
    preds_dict   = {}
    assets_dict  = {}

    for model_name in model_names:
        print(f"\n  Training {model_name}...")

        try:
            best_model, best_params, cv_mean, cv_std, cv_max_fold = optimize_with_optuna(
                X_tr_sc, y_tr, model_name, cv_folds=CV_FOLDS)

            best_model.fit(X_tr_sc, y_tr)
            y_pred = best_model.predict(X_te_sc)

            assets_dict[model_name] = {
                'model':             best_model,
                'scaler':            scaler,
                'best_params':       best_params,
                'cv_f1_mean':        cv_mean,
                'cv_f1_std':         cv_std,
                'cv_f1_best_fold':   cv_max_fold,
                'feature_names':     feature_names,
                'X_test_processed':  X_te_sc,
                'class_names':       class_labels
            }

            preds_dict[model_name] = {'y_true': y_te, 'y_pred': y_pred}

            acc    = accuracy_score(y_te, y_pred)
            f1_m   = f1_score(y_te, y_pred, average='macro',    zero_division=0)
            f1_w   = f1_score(y_te, y_pred, average='weighted', zero_division=0)
            prec_m = precision_score(y_te, y_pred, average='macro',    zero_division=0)
            rec_m  = recall_score(y_te, y_pred, average='macro',       zero_division=0)
            prec_w = precision_score(y_te, y_pred, average='weighted', zero_division=0)
            rec_w  = recall_score(y_te, y_pred, average='weighted',    zero_division=0)

            ci_seed = RANDOM_SEED   # seed único (alinhado com Sc3)
            print(f"  Calculating 95% bootstrap confidence intervals...")
            ci = calculate_confidence_intervals(y_te, y_pred, seed=ci_seed)

            results_dict[model_name] = {
                'accuracy_test':               acc,
                'accuracy_ci_lower':           ci['accuracy']['lower'],
                'accuracy_ci_upper':           ci['accuracy']['upper'],
                'f1_macro_test':               f1_m,
                'f1_macro_ci_lower':           ci['f1_macro']['lower'],
                'f1_macro_ci_upper':           ci['f1_macro']['upper'],
                'f1_weighted_test':            f1_w,
                'f1_weighted_ci_lower':        ci['f1_weighted']['lower'],
                'f1_weighted_ci_upper':        ci['f1_weighted']['upper'],
                'precision_macro_test':        prec_m,
                'precision_macro_ci_lower':    ci['precision_macro']['lower'],
                'precision_macro_ci_upper':    ci['precision_macro']['upper'],
                'recall_macro_test':           rec_m,
                'recall_macro_ci_lower':       ci['recall_macro']['lower'],
                'recall_macro_ci_upper':       ci['recall_macro']['upper'],
                'precision_weighted_test':     prec_w,
                'precision_weighted_ci_lower': ci['precision_weighted']['lower'],
                'precision_weighted_ci_upper': ci['precision_weighted']['upper'],
                'recall_weighted_test':        rec_w,
                'recall_weighted_ci_lower':    ci['recall_weighted']['lower'],
                'recall_weighted_ci_upper':    ci['recall_weighted']['upper'],
                'cv_f1_mean':                  cv_mean,
                'cv_f1_std':                   cv_std,
                'cv_f1_best_fold':             cv_max_fold,
                'best_params':                 best_params,
                # ── digits=4 para consistência com Scenarios 2 e 3 ──────────
                'classification_report':       classification_report(
                    y_te, y_pred, target_names=class_labels,
                    zero_division=0, digits=4),
            }

            print(f"  Accuracy : {acc:.4f} [{ci['accuracy']['lower']:.4f}, {ci['accuracy']['upper']:.4f}]")
            print(f"  F1-Macro : {f1_m:.4f} [{ci['f1_macro']['lower']:.4f}, {ci['f1_macro']['upper']:.4f}]")
            print(f"  F1-Weight: {f1_w:.4f} [{ci['f1_weighted']['lower']:.4f}, {ci['f1_weighted']['upper']:.4f}]")
            print(f"  CV F1    : {cv_mean:.4f} +/- {cv_std:.4f}  (best fold={cv_max_fold:.4f})")

        except Exception as e:
            print(f"  ERROR fitting {model_name}: {e}")
            import traceback; traceback.print_exc()
            results_dict[model_name] = None
            preds_dict[model_name]   = None
            assets_dict[model_name]  = None

    hp_dict = {m: a['best_params'] for m, a in assets_dict.items() if a}
    hp_file = os.path.join(hyperparams_folder, f'best_hyperparameters_{scenario_name}.json')
    with open(hp_file, 'w') as f:
        json.dump(hp_dict, f, indent=2, default=str)
    print(f"\n  Best hyperparameters saved: {hp_file}")

    return results_dict, preds_dict, assets_dict

# ====================================================================
# CONFUSION MATRIX PLOTTING
# ====================================================================

def plot_confusion_matrix_eval(y_true_cm, y_pred_cm, model_name_cm,
                                scenario_name_cm, class_labels_cm):
    if (y_true_cm is None or y_pred_cm is None
            or len(y_true_cm) == 0 or len(y_pred_cm) == 0):
        print(f"  Invalid data for CM — {model_name_cm} in {scenario_name_cm}. Skipping.")
        return

    cm      = confusion_matrix(y_true_cm, y_pred_cm)
    row_sum = cm.sum(axis=1)[:, np.newaxis]
    cm_pct  = np.zeros_like(cm, dtype=float)
    np.divide(cm.astype(float), row_sum, out=cm_pct, where=row_sum != 0)
    cm_pct *= 100

    annot = [[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"
              for j in range(cm.shape[1])] for i in range(cm.shape[0])]

    plt.figure(figsize=(max(8, len(class_labels_cm) * 1.8),
                        max(6, len(class_labels_cm) * 1.3)))
    sns.heatmap(cm, annot=annot, fmt='s', cmap='Blues',
                xticklabels=class_labels_cm, yticklabels=class_labels_cm,
                cbar_kws={'label': 'No. of Samples'})

    plt.title(f'Confusion Matrix — {model_name_cm}\n{scenario_name_cm}', fontsize=13)
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual',    fontsize=11)
    plt.xticks(fontsize=9, rotation=45, ha='right')
    plt.yticks(fontsize=9, rotation=0)

    acc_v  = accuracy_score(y_true_cm, y_pred_cm)
    f1_v   = f1_score(y_true_cm, y_pred_cm, average='macro',    zero_division=0)
    prec_v = precision_score(y_true_cm, y_pred_cm, average='macro', zero_division=0)
    rec_v  = recall_score(y_true_cm, y_pred_cm, average='macro',    zero_division=0)

    # digits=4 também no texto do gráfico para consistência
    plt.figtext(0.02, 0.01,
                f'Accuracy: {acc_v:.4f}\nF1-Macro: {f1_v:.4f}\n'
                f'Precision-Macro: {prec_v:.4f}\nRecall-Macro: {rec_v:.4f}',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgray', alpha=0.7))

    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    fname = (f'CM_{model_name_cm}_'
             f'{scenario_name_cm.replace(" ","_").replace("/","_")}.png')
    try:
        plt.savefig(os.path.join(plots_folder, fname), dpi=180, bbox_inches='tight')
        print(f"  Confusion matrix saved: {fname}")
    except Exception as e:
        print(f"  Error saving CM '{fname}': {e}")
    plt.close()

# ====================================================================
# SCENARIO EXECUTION
# ====================================================================

results_raw, preds_raw, models_raw = train_evaluate_models(
    X_train_flat, y_train_enc,
    X_test_flat,  y_test_enc,
    "Scenario1_Raw_Data", class_names
)

# McNemar's Test
if preds_raw and len([v for v in preds_raw.values() if v is not None]) > 1:
    mcnemar_df = perform_pairwise_mcnemar(preds_raw, y_test_enc, class_names)
    if not mcnemar_df.empty:
        mc_path = os.path.join(output_folder, 'mcnemar_pairwise_comparison.csv')
        mcnemar_df.to_csv(mc_path, index=False)
        print(f"\nMcNemar results saved: {mc_path}")

# SHAP Analysis
if APPLY_SHAP_ANALYSIS and SHAP_AVAILABLE and models_raw:
    first_assets = next((v for v in models_raw.values() if v), None)
    if first_assets and 'X_test_processed' in first_assets:
        perform_shap_analysis(
            models_raw,
            first_assets['X_test_processed'],
            first_assets['feature_names'],
            variables,
            "Scenario1_Raw_Data"
        )

# Confusion Matrices
if preds_raw:
    print("\nGenerating confusion matrices...")
    for mn, pd_ in preds_raw.items():
        if pd_ and 'y_true' in pd_ and 'y_pred' in pd_:
            plot_confusion_matrix_eval(
                pd_['y_true'], pd_['y_pred'],
                mn, "Scenario1_Raw_Data", class_names)

# ====================================================================
# FINAL REPORT
# ====================================================================

print("\n" + "="*70)
print(f"FINAL REPORT — {output_folder}")
print("="*70)

rows_report = []
for mn, res in results_raw.items():
    if res:
        rows_report.append({
            'Model':              mn,
            'Accuracy':           f"{res['accuracy_test']:.4f} [{res['accuracy_ci_lower']:.4f}, {res['accuracy_ci_upper']:.4f}]",
            'F1_Macro':           f"{res['f1_macro_test']:.4f} [{res['f1_macro_ci_lower']:.4f}, {res['f1_macro_ci_upper']:.4f}]",
            'F1_Weighted':        f"{res['f1_weighted_test']:.4f} [{res['f1_weighted_ci_lower']:.4f}, {res['f1_weighted_ci_upper']:.4f}]",
            'Precision_Macro':    f"{res['precision_macro_test']:.4f} [{res['precision_macro_ci_lower']:.4f}, {res['precision_macro_ci_upper']:.4f}]",
            'Recall_Macro':       f"{res['recall_macro_test']:.4f} [{res['recall_macro_ci_lower']:.4f}, {res['recall_macro_ci_upper']:.4f}]",
            'Precision_Weighted': f"{res['precision_weighted_test']:.4f}",
            'Recall_Weighted':    f"{res['recall_weighted_test']:.4f}",
            'CV_F1_Mean':         f"{res.get('cv_f1_mean',      float('nan')):.4f}",
            'CV_F1_Std':          f"{res.get('cv_f1_std',       float('nan')):.4f}",
            'CV_F1_Best_Fold':    f"{res.get('cv_f1_best_fold', float('nan')):.4f}",
        })
    else:
        rows_report.append({'Model': mn, 'Accuracy': 'Failed',
                            'F1_Macro': 'Failed', 'F1_Weighted': 'Failed',
                            'Precision_Macro': 'Failed', 'Recall_Macro': 'Failed',
                            'Precision_Weighted': 'Failed', 'Recall_Weighted': 'Failed',
                            'CV_F1_Mean': 'Failed', 'CV_F1_Std': 'Failed',
                            'CV_F1_Best_Fold': 'Failed'})

if rows_report:
    df_report = pd.DataFrame(rows_report)
    print("\nPerformance Summary (95% CI in brackets):")
    print(df_report.to_string(index=False))
    csv_path = os.path.join(output_folder, 'model_results.csv')
    df_report.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

valid = {k: v for k, v in results_raw.items() if v and 'f1_macro_test' in v}
if valid:
    best_model = max(valid, key=lambda k: valid[k]['f1_macro_test'])
    print(f"\nBEST RESULT (Test F1-Macro): {best_model} — {valid[best_model]['f1_macro_test']:.4f}")

print(f"\nSUMMARY:")
print(f"  Scenario     : Raw Sensor Data (no feature selection)")
print(f"  Train samples: {len(X_train_flat)}")
print(f"  Test samples : {len(X_test_flat)}")
print(f"  Raw features : {X_train_flat.shape[1]}")
print(f"  Models       : KNN, SVM, MLP, LightGBM")
print(f"  CV           : {CV_FOLDS}-fold Stratified (training data only)")
print(f"  Optuna       : TPE Sampler, {N_TRIALS_MAP} trials")
print(f"  Statistics   : McNemar test + Bootstrap 95% CI (1000 iterations)")
if APPLY_SHAP_ANALYSIS and SHAP_AVAILABLE:
    print(f"  SHAP         : variable importance — results in {shap_folder}")
print(f"  Hyperparams  : saved in {hyperparams_folder}")
print(f"\nREPRODUCIBILITY FIXES APPLIED:")
print(f"  FIX 1 - OMP_NUM_THREADS=1          : set before lightgbm import")
print(f"  ALIGNED - seed único = 42 para todos os modelos (alinhado com Sc3)")
print(f"  FIX 3 - random_state=seed           : SVM/MLP/LightGBM use model seed, not hardcoded 42")
print(f"  FIX 4 - deterministic=True          : LightGBM internal determinism")
print(f"  FIX 5 - bootstrap CI local RNG      : RandomState(seed) local, never touches global state")
print(f"  FIX 7 - SHAP sample seed            : local RandomState for reproducible sample selection")
print(f"  ALIGNED - Hyperspace with Scenario 3/4: KNN cosine+inv_dist, SVM rbf-only, MLP lbfgs+ARCH_MAP, LGB conservative")
print(f"  ALIGNED - CV: 10-fold (was 5-fold), n_jobs=1 in cross_val_score")
print(f"\nPROCESS FINISHED. Results in: {os.path.abspath(output_folder)}")