import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys

from scipy import stats
from scipy.stats import chi2

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             f1_score, precision_score, recall_score)

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings('ignore')

# ====================================================================
# KNN WEIGHT FUNCTIONS  (named — lambdas cannot be pickled by joblib)
# ====================================================================

def _knn_weight_inv_dist(distances):
    return 1.0 / (distances + 1e-6)

def _knn_weight_exp_dist(distances):
    return np.exp(-distances)

# ====================================================================
# REPRODUCIBILITY
# Estratégia: seed ÚNICO (RANDOM_SEED=42) para TODAS as combinações.
# Todas as otimizações Optuna usam TPESampler(seed=42), garantindo
# que a busca de hiperparâmetros é idêntica para todo grid point.
# Isso assegura comparação justa: diferenças de performance refletem
# apenas as features selecionadas, não o caminho de busca.
# ====================================================================
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED']   = str(RANDOM_SEED)
os.environ['OMP_NUM_THREADS']  = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

print(f"✓ Random seed fixo: {RANDOM_SEED} (igual para todas as combinações do grid)")
print(f"✓ OMP_NUM_THREADS=1  (determinismo LightGBM)")

optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.ioff()

script_version = "Scenario3_GridSearch_CorrxCumulative_FixedSeed"
print(f"✓ Starting {script_version}")
print("✓ SCENARIO 3: ReliefF + ANOVA + Grid Search over Correlation x Cumulative Thresholds")
print("✓ Seed único para todas as combinações — comparação justa entre grid points")
print("✓ None correlation threshold = no redundancy removal at all")

# ====================================================================
# CONFIGURATION
# ====================================================================
CORRELATION_THRESHOLDS_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, None]
CUMULATIVE_THRESHOLDS_TO_TEST  = [0.80, 0.85, 0.90, 0.95, 1.00]

SKIP_LIGHTGBM            = False
OPTIMIZE_HYPERPARAMETERS = True

RELIEFF_K_NEIGHBORS    = 15
RELIEFF_MAX_ITERATIONS = 201
RELIEFF_THRESHOLD      = 0.0
ANOVA_THRESHOLD        = 0.0
RELIEFF_WEIGHT         = 0.5
ANOVA_WEIGHT           = 0.5
MIN_FEATURES           = 5
MAX_FEATURES           = 144

DATASET_PATH       = './Preprocessed_Data'
OUTPUT_BASE_FOLDER = 'ML_Results_Scenario3_GridSearch'

N_TRIALS = 50   # 50 trials: adequado para dataset pequeno (201 amostras)
                # TPE converge em ~30-50 trials para espaços pequenos
                # 40 combos × 4 modelos × 50 trials × 10-fold CV = 80.000 fits

print(f"\n✓ Configuration:")
print(f"   Correlation thresholds : {CORRELATION_THRESHOLDS_TO_TEST}")
print(f"   Cumulative thresholds  : {CUMULATIVE_THRESHOLDS_TO_TEST}")
print(f"   Total grid points      : {len(CORRELATION_THRESHOLDS_TO_TEST) * len(CUMULATIVE_THRESHOLDS_TO_TEST)}")
print(f"   N_TRIALS per model     : {N_TRIALS} (seed fixo={RANDOM_SEED} para todos)")
print(f"   NOTE: None = no redundancy removal (all relevant features pass through)")

os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

# ====================================================================
# HELPERS
# ====================================================================

def _ct_label(ct):
    return 'NoCorr' if ct is None else f'{ct:.2f}'

def _ct_pct(ct):
    return 'NoCorr' if ct is None else str(int(ct * 100))


# ====================================================================
# MLP ARCHITECTURE MAP
# ====================================================================
MLP_ARCH_MAP = {
    '50':         (50,),
    '75':         (75,),
    '100':        (100,),
    #'125':        (125,),
    '150':        (150,),
    '200':        (200,),
    #'25_125':     (25, 125),
    '75_100':     (75, 100),
    '100_50':     (100, 50),
    #'125_200':    (125, 200),
    '150_75':     (150, 75),
    '200_150':    (200, 150),
}


# ====================================================================
# OPTUNA — seed único RANDOM_SEED para todas as combinações
# ====================================================================

def get_n_trials():
    return {'KNN': N_TRIALS, 'SVM': N_TRIALS, 'MLP': N_TRIALS, 'LightGBM': N_TRIALS}


def optimize_hyperparameters_optuna(X_train, y_train, model_name, feature_names=None):
    """Optuna TPE optimisation — seed fixo RANDOM_SEED para todos os grid points."""
    print(f"            Optimizing {model_name}...", end=" ")
    sys.stdout.flush()

    seed     = RANDOM_SEED   # mesmo seed para todas as combinações
    cv       = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    n_trials = get_n_trials()[model_name]

    X_input = (pd.DataFrame(X_train, columns=feature_names)
               if model_name == 'LightGBM' and feature_names is not None
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
            params = dict(
                C=trial.suggest_float('C', 0.1, 15, log=True),
                kernel=kernel,
                tol=1,
                class_weight=trial.suggest_categorical('class_weight', ['balanced', None]),
                probability=True,
                random_state=seed,
            )
            if kernel == 'rbf':
                gc = trial.suggest_categorical('gamma_choice', ['log_val'])
                params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log=True)

            model = SVC(**params)

        # ── MLP ──────────────────────────────────────────────────────────
        elif model_name == 'MLP':
            arch_map    = MLP_ARCH_MAP
            arch        = trial.suggest_categorical('hidden_layer_sizes_key', list(arch_map))
            solver      = trial.suggest_categorical('solver', ['lbfgs'])
            lr_schedule = trial.suggest_categorical('lr_schedule', ['invscale'])
            params = dict(
                hidden_layer_sizes=arch_map[arch],
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
            trial.set_user_attr('cv_scores', scores.tolist())
            trial.set_user_attr('cv_max',    float(scores.max()))
            trial.set_user_attr('cv_std',    float(scores.std()))
            return float(scores.mean())
        except Exception:
            return 0.0

    sampler = TPESampler(seed=seed)   # seed fixo para todos os grid points
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
        sp = dict(C=best_params['C'],
        kernel=best_params['kernel'],
        tol=1.0,
        class_weight=best_params['class_weight'],
        probability=True,
        random_state=seed)
        if best_params['kernel'] == 'rbf':
            sp['gamma'] = best_params['gamma']
        best_model = SVC(**sp)

    elif model_name == 'MLP':
        arch_map = MLP_ARCH_MAP
        solver = best_params['solver']
        mp = dict(
            hidden_layer_sizes=arch_map[best_params['hidden_layer_sizes_key']],
            activation=best_params['activation'], solver=solver,
            alpha=best_params['alpha'], max_iter=best_params['max_iter'],
            tol=1e-4, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=seed)
        if solver in ('adam', 'sgd'):
            mp['learning_rate_init'] = best_params['learning_rate_init']
            mp['batch_size']         = best_params['batch_size']
        if solver == 'adam':
            lr_sched = best_params.get('lr_schedule', 'constant')
            mp['learning_rate'] = lr_sched if lr_sched != 'invscaling' else 'constant'
            mp['beta_1']        = best_params['beta_1']
            mp['beta_2']        = best_params['beta_2']
        elif solver == 'sgd':
            mp['learning_rate'] = best_params.get('lr_schedule', 'constant')
            mp['momentum']      = best_params['momentum']
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

    # ── Recomputa CV no modelo reconstruído ──────────────────────────────────
    try:
        cv_rebuilt = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        X_rebuilt  = (pd.DataFrame(X_train, columns=feature_names)
                      if model_name == 'LightGBM' and feature_names is not None
                      else X_train)
        cv_scores_rebuilt = cross_val_score(best_model, X_rebuilt, y_train,
                                            cv=cv_rebuilt, scoring='f1_macro', n_jobs=1)
        _cv_mean     = float(cv_scores_rebuilt.mean())
        _cv_std      = float(cv_scores_rebuilt.std())
        _cv_max_fold = float(cv_scores_rebuilt.max())
        print(f"            {model_name} CV (rebuilt): mean={_cv_mean:.4f} ± {_cv_std:.4f}  "
              f"best_fold={_cv_max_fold:.4f}")
    except Exception:
        pass

    return best_model, best_params, _cv_mean, _cv_std, _cv_max_fold


# ====================================================================
# STATISTICAL HELPERS
# ====================================================================

def calculate_confidence_intervals(y_true, y_pred, n_bootstrap=1000,
                                   confidence=0.95, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)

    metrics = {k: [] for k in ['accuracy', 'f1_macro', 'f1_weighted',
                                'precision_macro', 'recall_macro',
                                'precision_weighted', 'recall_weighted']}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt  = y_true[idx]; yp = y_pred[idx]
        metrics['accuracy'].append(accuracy_score(yt, yp))
        metrics['f1_macro'].append(f1_score(yt, yp, average='macro', zero_division=0))
        metrics['f1_weighted'].append(f1_score(yt, yp, average='weighted', zero_division=0))
        metrics['precision_macro'].append(precision_score(yt, yp, average='macro', zero_division=0))
        metrics['recall_macro'].append(recall_score(yt, yp, average='macro', zero_division=0))
        metrics['precision_weighted'].append(precision_score(yt, yp, average='weighted', zero_division=0))
        metrics['recall_weighted'].append(recall_score(yt, yp, average='weighted', zero_division=0))

    alpha = 1 - confidence
    return {k: {'lower': np.percentile(v, alpha / 2 * 100),
                'upper': np.percentile(v, (1 - alpha / 2) * 100)}
            for k, v in metrics.items()}


def mcnemar_test(y_true, y_pred_a, y_pred_b, name_a, name_b):
    ca  = (np.array(y_pred_a) == np.array(y_true))
    cb  = (np.array(y_pred_b) == np.array(y_true))
    n01 = int(np.sum( ca & ~cb)); n10 = int(np.sum(~ca &  cb))
    n00 = int(np.sum(~ca & ~cb)); n11 = int(np.sum( ca &  cb))
    if (n01 + n10) == 0:
        stat, pval = 0.0, 1.0
    else:
        stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
        pval = 1 - chi2.cdf(stat, df=1)
    return {'model_a': name_a, 'model_b': name_b,
            'n01': n01, 'n10': n10, 'n00': n00, 'n11': n11,
            'statistic': float(stat), 'p_value': float(pval),
            'significant_at_0.05': pval < 0.05,
            'significant_at_0.01': pval < 0.01}


def perform_pairwise_mcnemar(predictions_dict, y_true):
    model_names = list(predictions_dict.keys())
    results = []
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1:]:
            if predictions_dict[ma] is None or predictions_dict[mb] is None:
                continue
            results.append(mcnemar_test(
                y_true,
                predictions_dict[ma]['y_pred'],
                predictions_dict[mb]['y_pred'],
                ma, mb))
    return pd.DataFrame(results) if results else pd.DataFrame()


# ====================================================================
# RELIEFF
# ====================================================================

class ReliefFSelector:
    def __init__(self, k_neighbors=15, max_iterations=201,
                 threshold=0.01, random_state=42):
        self.k              = k_neighbors
        self.max_iterations = max_iterations
        self.threshold      = threshold
        self.rng            = np.random.RandomState(random_state)
        self.weights        = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        feature_names = (X.columns.tolist() if isinstance(X, pd.DataFrame)
                         else [f"Feature_{i}" for i in range(n_features)])
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self.weights = {f: 0.0 for f in feature_names}
        Xn = (X_arr - X_arr.mean(axis=0)) / (X_arr.std(axis=0) + 1e-10)

        print(f"   Running ReliefF: k={self.k}, max_iter={self.max_iterations}")
        iters = min(self.max_iterations, n_samples)
        for _ in range(iters):
            idx  = self.rng.randint(0, n_samples)
            samp = Xn[idx]; cls = y[idx]
            same = np.where(y == cls)[0]; diff = np.where(y != cls)[0]
            if len(same) > 1:
                same = same[same != idx]
                hits = same[np.argsort(np.linalg.norm(Xn[same] - samp, axis=1))[:self.k]]
            else:
                hits = []
            if len(diff) > 0:
                miss = diff[np.argsort(np.linalg.norm(Xn[diff] - samp, axis=1))[:self.k]]
            else:
                miss = []
            for j, fn in enumerate(feature_names):
                dh = np.mean(np.abs(samp[j] - Xn[hits, j])) if len(hits) else 0
                dm = np.mean(np.abs(samp[j] - Xn[miss, j])) if len(miss) else 0
                self.weights[fn] += (dm - dh) / iters

        mn = min(self.weights.values()); mx = max(self.weights.values())
        if mx > mn:
            self.weights = {f: (w - mn) / (mx - mn) for f, w in self.weights.items()}
        print(f"      ReliefF done: "
              f"{len([w for w in self.weights.values() if w >= self.threshold])} features >= threshold")
        return self

    def get_scores(self):
        return self.weights


# ====================================================================
# ANOVA
# ====================================================================

def calculate_anova_scores(X, y):
    print(f"   Calculating ANOVA F-scores...")
    feature_names = (X.columns.tolist() if isinstance(X, pd.DataFrame)
                     else [f"Feature_{i}" for i in range(X.shape[1])])
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    f_scores = {}
    for i, fn in enumerate(feature_names):
        groups = [X_arr[y == c, i] for c in np.unique(y)]
        try:
            fs, _ = stats.f_oneway(*groups)
            f_scores[fn] = float(fs) if not np.isnan(fs) else 0.0
        except Exception:
            f_scores[fn] = 0.0
    print(f"      ANOVA done: "
          f"{len([s for s in f_scores.values() if s >= ANOVA_THRESHOLD])} features >= threshold")
    return f_scores


# ====================================================================
# STEP 1 — Compute ReliefF + ANOVA scores (runs ONCE)
# ====================================================================

def compute_base_scores(X_train, y_train):
    print(f"\n{'='*65}")
    print(f" STEP 1 — Computing ReliefF + ANOVA scores (runs ONCE)")
    print(f"{'='*65}")

    relieff = ReliefFSelector(k_neighbors=RELIEFF_K_NEIGHBORS,
                               max_iterations=RELIEFF_MAX_ITERATIONS,
                               threshold=RELIEFF_THRESHOLD,
                               random_state=RANDOM_SEED)
    relieff.fit(X_train, y_train)
    relieff_scores = relieff.get_scores()

    anova_scores = calculate_anova_scores(X_train, y_train)

    print(f"   Combining scores (ReliefFx{RELIEFF_WEIGHT} + ANOVAx{ANOVA_WEIGHT})...")
    combined = {}
    for f in relieff_scores:
        combined[f] = RELIEFF_WEIGHT * relieff_scores[f] + ANOVA_WEIGHT * anova_scores[f]
    mx = max(combined.values()) if combined else 1.0
    combined_scores = {f: s / mx for f, s in combined.items()} if mx > 0 else combined

    relevant_features = [f for f in combined_scores
                         if relieff_scores[f] >= RELIEFF_THRESHOLD
                         or anova_scores[f]   >= ANOVA_THRESHOLD]
    print(f"   Relevant features (either threshold): {len(relevant_features)}")

    features_sorted = sorted(
        [(f, combined_scores[f]) for f in relevant_features],
        key=lambda x: x[1], reverse=True)

    return dict(relieff_scores=relieff_scores,
                anova_scores=anova_scores,
                combined_scores=combined_scores,
                relevant_features=relevant_features,
                features_sorted=features_sorted)


# ====================================================================
# STEP 2 — Redundancy removal per CORRELATION_THRESHOLD
# ====================================================================

def remove_redundant_features(X, features, combined_scores, corr_threshold):
    X_sel  = X[features]
    corr   = X_sel.corr().abs()
    redund = set()
    fl     = list(features)
    for i in range(len(fl)):
        if fl[i] in redund: continue
        for j in range(i + 1, len(fl)):
            if fl[j] in redund: continue
            if corr.loc[fl[i], fl[j]] >= corr_threshold:
                si = combined_scores.get(fl[i], 0)
                sj = combined_scores.get(fl[j], 0)
                redund.add(fl[j] if si >= sj else fl[i])
    final = [f for f in fl if f not in redund]
    return final, redund


def compute_non_redundant_sets(X_train, base_scores):
    print(f"\n{'='*65}")
    print(f" STEP 2 — Redundancy removal (once per correlation threshold)")
    print(f"{'='*65}")

    relevant = base_scores['relevant_features']
    comb     = base_scores['combined_scores']
    nr_sets  = {}

    for ct in CORRELATION_THRESHOLDS_TO_TEST:
        label = _ct_label(ct)
        print(f"   Correlation threshold = {label}...", end=" ")

        if ct is None:
            non_red = list(relevant)
            redund  = set()
        elif len(relevant) > 1:
            non_red, redund = remove_redundant_features(X_train, relevant, comb, ct)
        else:
            non_red, redund = list(relevant), set()

        sorted_sub = sorted(
            [(f, comb[f]) for f in non_red],
            key=lambda x: x[1], reverse=True)

        nr_sets[ct] = dict(non_redundant=non_red,
                           redundant=redund,
                           features_sorted=sorted_sub)
        print(f"non-redundant={len(non_red)}  removed={len(redund)}")

    return nr_sets


# ====================================================================
# STEP 3 — Apply cumulative threshold cutoff
# ====================================================================

def apply_cumulative_threshold(nr_set, cumulative_threshold, combined_scores):
    features_sorted = nr_set['features_sorted']
    total   = sum(s for _, s in features_sorted)
    cumul   = 0.0
    selected = []
    for f, s in features_sorted:
        cumul += s
        selected.append(f)
        if total > 0 and cumul / total >= cumulative_threshold:
            break

    if len(selected) < MIN_FEATURES:
        selected = [f for f, _ in features_sorted[:MIN_FEATURES]]
    elif len(selected) > MAX_FEATURES:
        selected = [f for f, _ in features_sorted[:MAX_FEATURES]]

    return selected


# ====================================================================
# MODEL TRAINING + EVALUATION
# ====================================================================

def train_evaluate_models(X_train, y_train, X_test, y_test,
                          class_names, feature_names=None):
    model_names = ['KNN', 'SVM', 'MLP']
    if not SKIP_LIGHTGBM:
        model_names.append('LightGBM')

    results = {}; predictions = {}

    for mn in model_names:
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        try:
            model, best_params, cv_mean, cv_std, cv_max_fold = \
                optimize_hyperparameters_optuna(X_tr_sc, y_train, mn, feature_names)

            if mn == 'LightGBM' and feature_names is not None:
                Xtr = pd.DataFrame(X_tr_sc, columns=feature_names)
                Xte = pd.DataFrame(X_te_sc,  columns=feature_names)
                model.fit(Xtr, y_train); y_pred = model.predict(Xte)
            else:
                model.fit(X_tr_sc, y_train); y_pred = model.predict(X_te_sc)

            predictions[mn] = {'y_true': y_test, 'y_pred': y_pred}

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro',    zero_division=0)
            f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            prm = precision_score(y_test, y_pred, average='macro',    zero_division=0)
            rem = recall_score(y_test, y_pred, average='macro',       zero_division=0)
            prw = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rew = recall_score(y_test, y_pred, average='weighted',    zero_division=0)

            ci = calculate_confidence_intervals(y_test, y_pred, seed=RANDOM_SEED)

            print(f"            {mn:<10}  "
                  f"Test Acc={acc:.4f} [{ci['accuracy']['lower']:.4f}, {ci['accuracy']['upper']:.4f}]  "
                  f"Test F1={f1m:.4f} [{ci['f1_macro']['lower']:.4f}, {ci['f1_macro']['upper']:.4f}]")

            results[mn] = {
                'accuracy_test':               acc,
                'accuracy_ci_lower':           ci['accuracy']['lower'],
                'accuracy_ci_upper':           ci['accuracy']['upper'],
                'f1_macro_test':               f1m,
                'f1_macro_ci_lower':           ci['f1_macro']['lower'],
                'f1_macro_ci_upper':           ci['f1_macro']['upper'],
                'f1_weighted_test':            f1w,
                'f1_weighted_ci_lower':        ci['f1_weighted']['lower'],
                'f1_weighted_ci_upper':        ci['f1_weighted']['upper'],
                'precision_macro_test':        prm,
                'precision_macro_ci_lower':    ci['precision_macro']['lower'],
                'precision_macro_ci_upper':    ci['precision_macro']['upper'],
                'recall_macro_test':           rem,
                'recall_macro_ci_lower':       ci['recall_macro']['lower'],
                'recall_macro_ci_upper':       ci['recall_macro']['upper'],
                'precision_weighted_test':     prw,
                'precision_weighted_ci_lower': ci['precision_weighted']['lower'],
                'precision_weighted_ci_upper': ci['precision_weighted']['upper'],
                'recall_weighted_test':        rew,
                'recall_weighted_ci_lower':    ci['recall_weighted']['lower'],
                'recall_weighted_ci_upper':    ci['recall_weighted']['upper'],
                'cv_f1_mean':                  cv_mean,
                'cv_f1_std':                   cv_std,
                'cv_f1_best_fold':             cv_max_fold,
                'best_params':                 best_params,
                'classification_report':       classification_report(
                    y_test, y_pred, target_names=class_names, zero_division=0),
            }
        except Exception as e:
            print(f"  ERROR ({mn}): {e}")
            import traceback; traceback.print_exc()
            results[mn] = None; predictions[mn] = None

    return results, predictions


# ====================================================================
# VISUALIZATIONS
# ====================================================================

def plot_confusion_matrices(predictions, class_names, title, output_path):
    valid = {k: v for k, v in predictions.items() if v is not None}
    n = len(valid)
    if n == 0: return
    cols = min(n, 2); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i, (mname, pd_) in enumerate(valid.items()):
        ax = axes[i]
        cm = confusion_matrix(pd_['y_true'], pd_['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        acc = accuracy_score(pd_['y_true'], pd_['y_pred'])
        f1  = f1_score(pd_['y_true'], pd_['y_pred'], average='macro', zero_division=0)
        ax.set_title(f'{mname}  Acc={acc:.3f}  F1={f1:.3f}', fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close()


def plot_grid_heatmaps(grid_results, output_folder):
    print(f"\n✓ Generating grid heatmaps...")
    models  = ['KNN', 'SVM', 'MLP', 'LightGBM']
    metrics = [('accuracy_test',   'Accuracy'),
               ('f1_macro_test',   'F1-Macro (Test)'),
               ('cv_f1_mean',      'F1-Macro (CV Mean)'),
               ('cv_f1_best_fold', 'F1-Macro (CV Best Fold)')]

    corr_thresholds = CORRELATION_THRESHOLDS_TO_TEST
    cum_thresholds  = sorted(CUMULATIVE_THRESHOLDS_TO_TEST)
    corr_labels     = [_ct_label(ct) for ct in corr_thresholds]

    for model in models:
        fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
        fig.suptitle(f'Grid Search Heatmap — {model}', fontsize=14, fontweight='bold')

        for ax, (metric_key, metric_name) in zip(axes, metrics):
            matrix = pd.DataFrame(index=corr_labels,
                                   columns=[f'{t:.2f}' for t in cum_thresholds],
                                   dtype=float)
            for ct, ct_lbl in zip(corr_thresholds, corr_labels):
                for cmt in cum_thresholds:
                    r = grid_results.get((ct, cmt), {}).get(model)
                    matrix.loc[ct_lbl, f'{cmt:.2f}'] = (
                        r.get(metric_key, np.nan) if r else np.nan)

            sns.heatmap(matrix.astype(float), ax=ax, annot=True, fmt='.3f',
                        cmap='YlGnBu', vmin=0, vmax=1,
                        cbar_kws={'label': metric_name})
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Cumulative Threshold', fontsize=10)
            ax.set_ylabel('Correlation Threshold', fontsize=10)

        plt.tight_layout()
        path = os.path.join(output_folder, f'grid_heatmap_{model}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"   Saved: grid_heatmap_{model}.png")


def plot_feature_count_heatmap(feature_counts, output_folder):
    corr_thresholds = CORRELATION_THRESHOLDS_TO_TEST
    cum_thresholds  = sorted(CUMULATIVE_THRESHOLDS_TO_TEST)
    corr_labels     = [_ct_label(ct) for ct in corr_thresholds]

    matrix = pd.DataFrame(index=corr_labels,
                           columns=[f'{t:.2f}' for t in cum_thresholds],
                           dtype=float)
    for (ct, cmt), n in feature_counts.items():
        matrix.loc[_ct_label(ct), f'{cmt:.2f}'] = n

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix.astype(float), ax=ax, annot=True, fmt='.0f',
                cmap='Oranges', cbar_kws={'label': 'N features'})
    ax.set_title('Number of Selected Features per Grid Point',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Cumulative Threshold'); ax.set_ylabel('Correlation Threshold')
    plt.tight_layout()
    path = os.path.join(output_folder, 'grid_feature_counts.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: grid_feature_counts.png")


def plot_best_per_model(summary_df, output_folder):
    models  = summary_df['Model'].unique()
    metrics = ['F1_Macro', 'Accuracy', 'Precision_Macro', 'Recall_Macro', 'CV_F1_Best_Fold']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    fig.suptitle('Best Performance per Model (over all grid points)',
                 fontsize=13, fontweight='bold')

    for ax, metric in zip(axes, metrics):
        best = summary_df.groupby('Model')[metric].max().reindex(models)
        bars = ax.bar(best.index, best.values, color='steelblue', alpha=0.85)
        ax.set_title(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric); ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, best.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(output_folder, 'best_per_model.png')
    plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: best_per_model.png")


# ====================================================================
# SENSITIVITY ANALYSIS PLOTS
# ====================================================================

def plot_selected_feature_importance(selected_features, base_scores, combo_label,
                                     combo_folder, corr_threshold, cum_threshold):
    if not selected_features:
        return

    relieff  = base_scores['relieff_scores']
    anova    = base_scores['anova_scores']
    combined = base_scores['combined_scores']

    rf_vals = np.array([relieff.get(f,  0.0) for f in selected_features])
    an_vals = np.array([anova.get(f,    0.0) for f in selected_features])
    co_vals = np.array([combined.get(f, 0.0) for f in selected_features])

    def _norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn) if mx > mn else np.ones_like(arr)

    rf_n = _norm(rf_vals)
    an_n = _norm(an_vals)
    co_n = _norm(co_vals)

    n      = len(selected_features)
    x      = np.arange(n)
    width  = 0.25
    labels = [f[:18] + '…' if len(f) > 18 else f for f in selected_features]

    fig_w  = max(14, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    ax.bar(x - width, rf_n, width, label='ReliefF (norm)', color='#FFA500')
    ax.bar(x,         an_n, width, label='ANOVA (norm)',    color='#2ca02c')
    ax.bar(x + width, co_n, width, label='Combined (norm)', color='#1f77b4')

    ct_str = _ct_label(corr_threshold)
    ax.set_title(
        f'Feature Importance — Selected Features\n'
        f'Corr={ct_str}  |  Cum={cum_threshold:.2f}  '
        f'|  {n} features  [{combo_label}]',
        fontsize=12, fontweight='bold')
    ax.set_xlabel('Features',         fontsize=10)
    ax.set_ylabel('Normalized Score', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    out = os.path.join(combo_folder, 'feature_importance_comparison.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()


def plot_sensitivity_analysis(summary_df, feature_counts, output_folder):
    print(f"\n✓ Generating sensitivity analysis plots...")

    models       = ['KNN', 'SVM', 'MLP', 'LightGBM']
    metrics      = [('Accuracy', 'Accuracy'), ('F1_Macro', 'F1-Macro')]
    cum_vals     = sorted(summary_df['Cum_Threshold'].unique())

    corr_order   = [_ct_label(ct) for ct in CORRELATION_THRESHOLDS_TO_TEST]
    n_corr       = len(corr_order)
    n_cum        = len(cum_vals)
    cum_colors   = plt.cm.viridis(np.linspace(0.2, 0.9, n_cum))
    corr_colors  = plt.cm.plasma(np.linspace(0.2, 0.9, n_corr))
    corr_pos     = {lbl: i for i, lbl in enumerate(corr_order)}

    # FIG 1: Métrica vs Correlation Threshold
    fig, axes = plt.subplots(len(metrics), len(models),
                             figsize=(5 * len(models), 4.5 * len(metrics)),
                             sharey='row')
    fig.suptitle('Sensitivity Analysis — Metric vs Correlation Threshold\n'
                 '(each line = one Cumulative Threshold)',
                 fontsize=14, fontweight='bold')

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, model in enumerate(models):
            ax  = axes[row, col]
            sub = summary_df[summary_df['Model'] == model].copy()

            for ci, cum in enumerate(cum_vals):
                data = sub[sub['Cum_Threshold'] == cum].copy()
                data['_pos'] = data['Corr_Threshold'].map(corr_pos)
                data = data.sort_values('_pos')
                ax.plot(data['_pos'], data[metric_col],
                        marker='o', color=cum_colors[ci], linewidth=2,
                        label=f'Cum={cum:.2f}')
                lo_col = f'{metric_col}_CI_Lower'
                hi_col = f'{metric_col}_CI_Upper'
                if lo_col in sub.columns and hi_col in sub.columns:
                    ax.fill_between(data['_pos'], data[lo_col], data[hi_col],
                                    alpha=0.08, color=cum_colors[ci])

            ax.set_title(f'{model}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Correlation Threshold', fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(list(range(n_corr)))
            ax.set_xticklabels(corr_order, rotation=45, fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='lower left')

    plt.tight_layout()
    p = os.path.join(output_folder, 'sensitivity_corr_threshold.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: sensitivity_corr_threshold.png")

    # FIG 2: Métrica vs Cumulative Threshold
    fig, axes = plt.subplots(len(metrics), len(models),
                             figsize=(5 * len(models), 4.5 * len(metrics)),
                             sharey='row')
    fig.suptitle('Sensitivity Analysis — Metric vs Cumulative Threshold\n'
                 '(each line = one Correlation Threshold)',
                 fontsize=14, fontweight='bold')

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, model in enumerate(models):
            ax  = axes[row, col]
            sub = summary_df[summary_df['Model'] == model].copy()

            for ci, ct_lbl in enumerate(corr_order):
                data = (sub[sub['Corr_Threshold'] == ct_lbl]
                        .sort_values('Cum_Threshold'))
                ax.plot(data['Cum_Threshold'], data[metric_col],
                        marker='s', color=corr_colors[ci], linewidth=2,
                        label=f'Corr={ct_lbl}')
                lo_col = f'{metric_col}_CI_Lower'
                hi_col = f'{metric_col}_CI_Upper'
                if lo_col in sub.columns and hi_col in sub.columns:
                    ax.fill_between(data['Cum_Threshold'], data[lo_col], data[hi_col],
                                    alpha=0.08, color=corr_colors[ci])

            ax.set_title(f'{model}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Cumulative Threshold', fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(cum_vals)
            ax.set_xticklabels([f'{v:.2f}' for v in cum_vals], rotation=45, fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    p = os.path.join(output_folder, 'sensitivity_cum_threshold.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: sensitivity_cum_threshold.png")

    # FIG 3: Métrica vs Nº de Features
    if 'Num_Features' not in summary_df.columns:
        summary_df['Num_Features'] = summary_df.apply(
            lambda r: feature_counts.get(
                next((k for k in feature_counts
                      if _ct_label(k[0]) == r['Corr_Threshold']
                      and k[1] == r['Cum_Threshold']), None),
                np.nan),
            axis=1)

    fig, axes = plt.subplots(len(metrics), len(models),
                             figsize=(5 * len(models), 4.5 * len(metrics)),
                             sharey='row')
    fig.suptitle('Sensitivity Analysis — Metric vs Number of Selected Features',
                 fontsize=14, fontweight='bold')

    for row, (metric_col, metric_label) in enumerate(metrics):
        for col, model in enumerate(models):
            ax  = axes[row, col]
            sub = summary_df[summary_df['Model'] == model].dropna(
                subset=['Num_Features', metric_col])

            sc = ax.scatter(sub['Num_Features'], sub[metric_col],
                            c=sub['Num_Features'], cmap='plasma',
                            s=60, alpha=0.8, zorder=3)

            if len(sub) > 1:
                z  = np.polyfit(sub['Num_Features'], sub[metric_col], 1)
                p_ = np.poly1d(z)
                xs = np.linspace(sub['Num_Features'].min(),
                                 sub['Num_Features'].max(), 100)
                ax.plot(xs, p_(xs), 'k--', linewidth=1.2, alpha=0.6,
                        label=f'trend  slope={z[0]:.4f}')
                ax.legend(fontsize=7)

            plt.colorbar(sc, ax=ax, label='N features', pad=0.02)
            ax.set_title(f'{model}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Number of Selected Features', fontsize=9)
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_folder, 'sensitivity_n_features.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: sensitivity_n_features.png")

    # FIG 4: Agregado mean ± std entre modelos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sensitivity Analysis — Aggregated View (mean ± std across models)',
                 fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for ci, cum in enumerate(cum_vals):
        grp = summary_df[summary_df['Cum_Threshold'] == cum].copy()
        grp['_pos'] = grp['Corr_Threshold'].map(corr_pos)
        grp = grp.groupby('_pos')['F1_Macro'].agg(['mean', 'std']).reset_index()
        ax.plot(grp['_pos'], grp['mean'], marker='o',
                color=cum_colors[ci], linewidth=2, label=f'Cum={cum:.2f}')
        ax.fill_between(grp['_pos'], grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'], alpha=0.15, color=cum_colors[ci])
    ax.set_title('F1-Macro vs Corr Threshold\n(mean ± std over models)', fontweight='bold')
    ax.set_xlabel('Correlation Threshold'); ax.set_ylabel('F1-Macro')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xticks(list(range(n_corr)))
    ax.set_xticklabels(corr_order, rotation=45)

    ax = axes[0, 1]
    for ci, ct_lbl in enumerate(corr_order):
        grp = (summary_df[summary_df['Corr_Threshold'] == ct_lbl]
               .groupby('Cum_Threshold')['F1_Macro']
               .agg(['mean', 'std']).reset_index())
        ax.plot(grp['Cum_Threshold'], grp['mean'], marker='s',
                color=corr_colors[ci], linewidth=2, label=f'Corr={ct_lbl}')
        ax.fill_between(grp['Cum_Threshold'], grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'], alpha=0.15, color=corr_colors[ci])
    ax.set_title('F1-Macro vs Cum Threshold\n(mean ± std over models)', fontweight='bold')
    ax.set_xlabel('Cumulative Threshold'); ax.set_ylabel('F1-Macro')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xticks(cum_vals)
    ax.set_xticklabels([f'{v:.2f}' for v in cum_vals], rotation=45)

    ax = axes[1, 0]
    for ci, cum in enumerate(cum_vals):
        grp = summary_df[summary_df['Cum_Threshold'] == cum].copy()
        grp['_pos'] = grp['Corr_Threshold'].map(corr_pos)
        grp = grp.groupby('_pos')['Accuracy'].agg(['mean', 'std']).reset_index()
        ax.plot(grp['_pos'], grp['mean'], marker='o',
                color=cum_colors[ci], linewidth=2, label=f'Cum={cum:.2f}')
        ax.fill_between(grp['_pos'], grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'], alpha=0.15, color=cum_colors[ci])
    ax.set_title('Accuracy vs Corr Threshold\n(mean ± std over models)', fontweight='bold')
    ax.set_xlabel('Correlation Threshold'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xticks(list(range(n_corr)))
    ax.set_xticklabels(corr_order, rotation=45)

    ax = axes[1, 1]
    for ci, ct_lbl in enumerate(corr_order):
        grp = (summary_df[summary_df['Corr_Threshold'] == ct_lbl]
               .groupby('Cum_Threshold')['Accuracy']
               .agg(['mean', 'std']).reset_index())
        ax.plot(grp['Cum_Threshold'], grp['mean'], marker='s',
                color=corr_colors[ci], linewidth=2, label=f'Corr={ct_lbl}')
        ax.fill_between(grp['Cum_Threshold'], grp['mean'] - grp['std'],
                        grp['mean'] + grp['std'], alpha=0.15, color=corr_colors[ci])
    ax.set_title('Accuracy vs Cum Threshold\n(mean ± std over models)', fontweight='bold')
    ax.set_xlabel('Cumulative Threshold'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xticks(cum_vals)
    ax.set_xticklabels([f'{v:.2f}' for v in cum_vals], rotation=45)

    plt.tight_layout()
    p = os.path.join(output_folder, 'sensitivity_combined.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"   Saved: sensitivity_combined.png")


# ====================================================================
# MAIN
# ====================================================================

def main():
    print(f"\n{'='*70}")
    print(f" SCENARIO 3  —  GRID SEARCH: Correlation x Cumulative Thresholds")
    print(f" Seed fixo: {RANDOM_SEED} para todas as combinações | {N_TRIALS} trials/modelo")
    print(f"{'='*70}")

    print(f"\n✓ Loading datasets from {DATASET_PATH}...")
    train_file = os.path.join(DATASET_PATH, 'dataset_train_transformed.csv')
    test_file  = os.path.join(DATASET_PATH, 'dataset_test_transformed.csv')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("ERROR: Dataset files not found!"); return

    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)
    print(f"   Train: {train_df.shape}  |  Test: {test_df.shape}")

    X_train_full = train_df.drop('label', axis=1)
    X_test_full  = test_df.drop('label',  axis=1)

    le          = LabelEncoder()
    y_train     = le.fit_transform(train_df['label'])
    y_test      = le.transform(test_df['label'])
    class_names = list(le.classes_)
    print(f"   Features: {X_train_full.shape[1]}  |  Classes: {class_names}")

    base_scores = compute_base_scores(X_train_full, y_train)

    pd.DataFrame({
        'Feature':        list(base_scores['combined_scores'].keys()),
        'ReliefF_Score':  [base_scores['relieff_scores'][f]  for f in base_scores['combined_scores']],
        'ANOVA_Score':    [base_scores['anova_scores'][f]    for f in base_scores['combined_scores']],
        'Combined_Score': list(base_scores['combined_scores'].values()),
    }).sort_values('Combined_Score', ascending=False).to_csv(
        os.path.join(OUTPUT_BASE_FOLDER, 'global_feature_scores.csv'), index=False)
    print("✓ Global feature scores saved: global_feature_scores.csv")

    nr_sets = compute_non_redundant_sets(X_train_full, base_scores)

    print(f"\n{'='*70}")
    print(f" STEP 3+4 — Grid loop ({len(CORRELATION_THRESHOLDS_TO_TEST)}x"
          f"{len(CUMULATIVE_THRESHOLDS_TO_TEST)} = "
          f"{len(CORRELATION_THRESHOLDS_TO_TEST) * len(CUMULATIVE_THRESHOLDS_TO_TEST)} combinations)")
    print(f"{'='*70}")

    grid_results   = {}
    grid_preds     = {}
    feature_counts = {}

    total_combos = len(CORRELATION_THRESHOLDS_TO_TEST) * len(CUMULATIVE_THRESHOLDS_TO_TEST)
    combo_idx    = 0

    for ct in CORRELATION_THRESHOLDS_TO_TEST:
        for cmt in CUMULATIVE_THRESHOLDS_TO_TEST:
            combo_idx   += 1
            key          = (ct, cmt)
            ct_pct_str   = _ct_pct(ct)
            cmt_pct      = int(cmt * 100)
            combo_label  = f'Corr{ct_pct_str}_Cum{cmt_pct}'
            combo_folder = os.path.join(OUTPUT_BASE_FOLDER, combo_label)
            os.makedirs(combo_folder, exist_ok=True)

            ct_str = _ct_label(ct)
            print(f"\n{'─'*65}")
            print(f" Combo {combo_idx}/{total_combos} "
                  f"| Corr={ct_str}  Cum={cmt:.2f}  [{combo_label}]  seed={RANDOM_SEED}")
            print(f"{'─'*65}")

            selected = apply_cumulative_threshold(
                nr_sets[ct], cmt, base_scores['combined_scores'])
            feature_counts[key] = len(selected)
            print(f"   Selected features: {len(selected)}")

            with open(os.path.join(combo_folder, 'selected_features.json'), 'w') as f:
                json.dump({'corr_threshold': ct_str,
                           'cum_threshold':  cmt,
                           'n_features':     len(selected),
                           'features':       selected}, f, indent=2)

            plot_selected_feature_importance(
                selected, base_scores, combo_label,
                combo_folder, ct, cmt)

            X_tr = X_train_full[selected]
            X_te = X_test_full[selected]

            results, predictions = train_evaluate_models(
                X_tr, y_train, X_te, y_test, class_names, selected)

            grid_results[key] = results
            grid_preds[key]   = predictions

            valid_p = {k: v for k, v in predictions.items() if v is not None}
            if len(valid_p) > 1:
                mc_df = perform_pairwise_mcnemar(valid_p, y_test)
                if not mc_df.empty:
                    mc_df.to_csv(os.path.join(combo_folder, 'mcnemar.csv'), index=False)

            rows = []
            for mn, res in results.items():
                if res is None: continue
                rows.append({
                    'Model':                    mn,
                    'Corr_Threshold':           ct_str,
                    'Cum_Threshold':            cmt,
                    'Num_Features':             len(selected),
                    'Accuracy':                 res['accuracy_test'],
                    'Accuracy_CI_Lower':        res['accuracy_ci_lower'],
                    'Accuracy_CI_Upper':        res['accuracy_ci_upper'],
                    'F1_Macro':                 res['f1_macro_test'],
                    'F1_Macro_CI_Lower':        res['f1_macro_ci_lower'],
                    'F1_Macro_CI_Upper':        res['f1_macro_ci_upper'],
                    'F1_Weighted':              res['f1_weighted_test'],
                    'F1_Weighted_CI_Lower':     res['f1_weighted_ci_lower'],
                    'F1_Weighted_CI_Upper':     res['f1_weighted_ci_upper'],
                    'Precision_Macro':          res['precision_macro_test'],
                    'Precision_Macro_CI_Lower': res['precision_macro_ci_lower'],
                    'Precision_Macro_CI_Upper': res['precision_macro_ci_upper'],
                    'Recall_Macro':             res['recall_macro_test'],
                    'Recall_Macro_CI_Lower':    res['recall_macro_ci_lower'],
                    'Recall_Macro_CI_Upper':    res['recall_macro_ci_upper'],
                    'CV_F1_Mean':               res.get('cv_f1_mean',      float('nan')),
                    'CV_F1_Std':                res.get('cv_f1_std',       float('nan')),
                    'CV_F1_Best_Fold':          res.get('cv_f1_best_fold', float('nan')),
                    'Best_Params':              str(res['best_params']),
                })
            pd.DataFrame(rows).to_csv(
                os.path.join(combo_folder, 'model_results.csv'), index=False)

            with open(os.path.join(combo_folder, 'best_hyperparameters.json'), 'w') as f:
                json.dump({m: r['best_params'] for m, r in results.items() if r},
                          f, indent=2, default=str)

            plot_confusion_matrices(
                predictions, class_names,
                f'Confusion Matrices — Corr={ct_str}  Cum={cmt:.2f}',
                os.path.join(combo_folder, 'confusion_matrices.png'))

    # ── Aggregate summary CSV ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" BUILDING AGGREGATE SUMMARY")
    print(f"{'='*70}")

    all_rows = []
    for (ct, cmt), res_dict in grid_results.items():
        ct_str = _ct_label(ct)
        for mn, res in res_dict.items():
            if res is None: continue
            all_rows.append({
                'Corr_Threshold':           ct_str,
                'Cum_Threshold':            cmt,
                'Combo_Label':              f'Corr{_ct_pct(ct)}_Cum{int(cmt*100)}',
                'Model':                    mn,
                'Num_Features':             feature_counts[(ct, cmt)],
                'Accuracy':                 res['accuracy_test'],
                'Accuracy_CI_Lower':        res['accuracy_ci_lower'],
                'Accuracy_CI_Upper':        res['accuracy_ci_upper'],
                'F1_Macro':                 res['f1_macro_test'],
                'F1_Macro_CI_Lower':        res['f1_macro_ci_lower'],
                'F1_Macro_CI_Upper':        res['f1_macro_ci_upper'],
                'F1_Weighted':              res['f1_weighted_test'],
                'F1_Weighted_CI_Lower':     res['f1_weighted_ci_lower'],
                'F1_Weighted_CI_Upper':     res['f1_weighted_ci_upper'],
                'Precision_Macro':          res['precision_macro_test'],
                'Precision_Macro_CI_Lower': res['precision_macro_ci_lower'],
                'Precision_Macro_CI_Upper': res['precision_macro_ci_upper'],
                'Recall_Macro':             res['recall_macro_test'],
                'Recall_Macro_CI_Lower':    res['recall_macro_ci_lower'],
                'Recall_Macro_CI_Upper':    res['recall_macro_ci_upper'],
                'CV_F1_Mean':               res.get('cv_f1_mean',      float('nan')),
                'CV_F1_Std':                res.get('cv_f1_std',       float('nan')),
                'CV_F1_Best_Fold':          res.get('cv_f1_best_fold', float('nan')),
            })

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(OUTPUT_BASE_FOLDER, 'grid_search_summary.csv'),
                      index=False, float_format='%.6f')
    print(f"✓ Summary saved: grid_search_summary.csv  ({len(summary_df)} rows)")

    plot_grid_heatmaps(grid_results, OUTPUT_BASE_FOLDER)
    plot_feature_count_heatmap(feature_counts, OUTPUT_BASE_FOLDER)
    plot_best_per_model(summary_df, OUTPUT_BASE_FOLDER)
    plot_sensitivity_analysis(summary_df, feature_counts, OUTPUT_BASE_FOLDER)

    # ── Best configurations ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f" BEST CONFIGURATION PER MODEL (by F1-Macro)")
    print(f"{'='*70}")
    for model in ['KNN', 'SVM', 'MLP', 'LightGBM']:
        sub = summary_df[summary_df['Model'] == model]
        if sub.empty: continue
        best         = sub.loc[sub['F1_Macro'].idxmax()]
        ct_str       = best['Corr_Threshold']
        cmt          = best['Cum_Threshold']
        orig_ct      = None if ct_str == 'NoCorr' else float(ct_str)
        res          = grid_results.get((orig_ct, cmt), {}).get(model, {}) or {}
        cv_mean      = res.get('cv_f1_mean',      float('nan'))
        cv_std       = res.get('cv_f1_std',       float('nan'))
        cv_best_fold = res.get('cv_f1_best_fold', float('nan'))
        print(f"\n  {model}:")
        print(f"    Test Accuracy = {best['Accuracy']:.4f} "
              f"[{best['Accuracy_CI_Lower']:.4f}, {best['Accuracy_CI_Upper']:.4f}]")
        print(f"    Test F1-Macro = {best['F1_Macro']:.4f} "
              f"[{best['F1_Macro_CI_Lower']:.4f}, {best['F1_Macro_CI_Upper']:.4f}]")
        if not np.isnan(cv_mean):
            print(f"    CV   F1-Macro = {cv_mean:.4f} ± {cv_std:.4f}  "
                  f"(best fold = {cv_best_fold:.4f})")
        print(f"    Corr = {ct_str} | Cumul = {int(cmt*100)}% | Features = {int(best['Num_Features'])}")
        print(f"    Params = {str(res.get('best_params', 'n/a'))[:120]}...")

    print(f"\n{'='*70}")
    print(f" SCENARIO 3 GRID SEARCH COMPLETED")
    print(f"{'='*70}")
    print(f" Results folder : {OUTPUT_BASE_FOLDER}/")
    print(f"\n✓ Reproducibility summary:")
    print(f"   Seed único {RANDOM_SEED} para todas as {total_combos} combinações do grid")
    print(f"   OMP_NUM_THREADS=1     : determinismo LightGBM")
    print(f"   deterministic=True    : LightGBM modo determinístico interno")
    print(f"   RandomState local     : bootstrap CI isolado do estado global")
    print(f"   {N_TRIALS} trials/modelo  : adequado para dataset de {train_df.shape[0]} amostras")


if __name__ == "__main__":
    main()