import os
import random

os.environ['OMP_NUM_THREADS']  = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, classification_report)
import lightgbm as lgb
import shap
import warnings

warnings.filterwarnings('ignore')
plt.ioff()

# ====================================================================
# RANDOM SEED — idêntico ao Scenario 3 (RANDOM_SEED = 42)
# ====================================================================
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

print(f"✓ RANDOM_SEED = {RANDOM_SEED}  (idêntico ao Scenario 3)")
print(f"✓ OMP_NUM_THREADS=1  (determinismo LightGBM)")

# ====================================================================
# CONFIGURATION
# ====================================================================
# Combo do Sc3 equivalente a 144 features sem filtros:
#   Corr=None → label 'NoCorr'  |  Cum=1.00 → 100%
#   Pasta gerada pelo Sc3: CorrNoCorr_Cum100
SC3_RESULTS  = './ML_Results_Scenario3_GridSearch'
DATASET_PATH = './Preprocessed_Data'
COMBO_FOLDER = os.path.join(SC3_RESULTS, 'CorrNoCorr_Cum100')

output_folder = 'ML_Results_Scenario2_SHAP'
os.makedirs(output_folder, exist_ok=True)
shap_folder = os.path.join(output_folder, 'SHAP_Analysis')
os.makedirs(shap_folder, exist_ok=True)
plots_folder = os.path.join(output_folder, 'Plots')
os.makedirs(plots_folder, exist_ok=True)

print(f"✓ Loading hyperparameters from: {COMBO_FOLDER}")
print(f"✓ Output folder: {output_folder}")

# ====================================================================
# KNN WEIGHT FUNCTIONS  (named — lambdas cannot be pickled by joblib)
# ====================================================================

def _knn_weight_inv_dist(distances):
    return 1.0 / (distances + 1e-6)

def _knn_weight_exp_dist(distances):
    return np.exp(-distances)

# ====================================================================
# MLP ARCHITECTURE MAP
# Idêntico ao Scenario 3 — NÃO adicionar arquiteturas extras aqui.
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

# ====================================================================
# LOAD DATASET
# ====================================================================

def load_dataset():
    print(f"\nLoading dataset...")
    train_file = os.path.join(DATASET_PATH, 'dataset_train_transformed.csv')
    test_file  = os.path.join(DATASET_PATH, 'dataset_test_transformed.csv')

    train_df = pd.read_csv(train_file)
    test_df  = pd.read_csv(test_file)

    X_train = train_df.drop('label', axis=1)
    X_test  = test_df.drop('label',  axis=1)

    le          = LabelEncoder()
    y_train     = le.fit_transform(train_df['label'])
    y_test      = le.transform(test_df['label'])
    class_names = list(le.classes_)

    print(f"✓ Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"✓ Classes: {class_names}")
    return X_train, X_test, y_train, y_test, class_names

# ====================================================================
# LOAD SELECTED FEATURES FROM COMBO
# ====================================================================

def load_selected_features():
    feat_file = os.path.join(COMBO_FOLDER, 'selected_features.json')
    if not os.path.exists(feat_file):
        print(f"ERROR: {feat_file} not found — verifique o caminho COMBO_FOLDER")
        return None
    with open(feat_file) as f:
        data = json.load(f)
    features = data['features']
    print(f"✓ Features carregadas do Sc3 combo: {len(features)}")
    return features

# ====================================================================
# REBUILD MODELS FROM SAVED HYPERPARAMETERS
# Lógica de reconstrução idêntica ao Scenario 3 para cada modelo.
# ====================================================================

def rebuild_models(best_params_path):
    print(f"\nReconstruindo modelos a partir de: {best_params_path}")
    with open(best_params_path) as f:
        all_params = json.load(f)

    seed   = RANDOM_SEED   # mesmo seed do Sc3
    models = {}

    for model_name, best_params in all_params.items():
        print(f"  Reconstruindo {model_name}...")
        try:

            # ── KNN ───────────────────────────────────────────────────────
            if model_name == 'KNN':
                wc = best_params['weights']
                fw = (_knn_weight_inv_dist if wc == 'inv_dist'
                      else (_knn_weight_exp_dist if wc == 'exp_dist' else wc))
                mt = best_params['metric']
                from sklearn.neighbors import KNeighborsClassifier
                models[model_name] = KNeighborsClassifier(
                    n_neighbors=best_params['n_neighbors'],
                    weights=fw,
                    metric=mt,
                    p=best_params['p'],
                    algorithm='brute' if mt == 'cosine' else 'kd_tree')

            # ── SVM ───────────────────────────────────────────────────────
            # Sc3: kernel fixo em 'rbf', gamma_choice fixo em 'scale'.
            # O Optuna salva 'gamma' como float (log_val), mas gamma_choice='scale'
            # significa que o modelo usa gamma='scale' — o float não é usado.
            # Reproduzimos exatamente isso aqui.
            elif model_name == 'SVM':
                from sklearn.svm import SVC
                sp = dict(
                    C=best_params['C'],
                    kernel=best_params['kernel'],
                    class_weight=best_params['class_weight'],
                    probability=True,
                    random_state=seed)
                gc = best_params.get('gamma_choice', 'scale')
                if gc == 'log_val':
                    sp['gamma'] = best_params['gamma']  # usa o float salvo
                else:
                    sp['gamma'] = gc
                models[model_name] = SVC(**sp)
            # ── MLP ───────────────────────────────────────────────────────
            # Sc3: solver fixo em 'lbfgs', lr_schedule fixo em 'invscale'.
            # Como lbfgs não usa learning_rate_init/batch_size/beta, esses
            # parâmetros não aparecem no espaço do Sc3 e não são reconstruídos.
            # validation_fraction=0.2 (igual ao Sc3 no rebuild).
            elif model_name == 'MLP':
                from sklearn.neural_network import MLPClassifier
                mp = dict(
                    hidden_layer_sizes=MLP_ARCH_MAP[best_params['hidden_layer_sizes_key']],
                    activation=best_params['activation'],
                    solver=best_params['solver'],        # 'lbfgs' no Sc3
                    alpha=best_params['alpha'],
                    max_iter=best_params['max_iter'],
                    tol=1e-4,
                    early_stopping=True,
                    validation_fraction=0.2,             # idêntico ao Sc3 rebuild
                    n_iter_no_change=20,
                    random_state=seed)
                # lbfgs não usa learning_rate_init, batch_size, beta_1, beta_2,
                # momentum — não adicionamos nada além do que está acima.
                models[model_name] = MLPClassifier(**mp)

            # ── LightGBM ──────────────────────────────────────────────────
            elif model_name == 'LightGBM':
                lgb_keys = ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth',
                            'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
                            'min_child_samples', 'min_data_in_leaf',
                            'feature_fraction', 'bagging_fraction', 'bagging_freq']
                lp = {k: best_params[k] for k in lgb_keys if k in best_params}
                lp.update(
                    class_weight='balanced',
                    verbosity=-1,
                    n_jobs=1,
                    force_col_wise=True,
                    random_state=seed,
                    deterministic=True,
                    force_row_wise=False)
                models[model_name] = lgb.LGBMClassifier(**lp)

            print(f"    ✓ {model_name} reconstruído")

        except Exception as e:
            print(f"    ERRO ao reconstruir {model_name}: {e}")
            import traceback; traceback.print_exc()

    return models

# ====================================================================
# TRAIN MODELS ON FULL TRAIN SET AND EVALUATE
# ====================================================================

def train_and_evaluate(models, X_train_sc, X_test_sc, y_train, y_test,
                       class_names, feature_names):
    print(f"\nFitando modelos no conjunto de treino completo...")
    trained = {}
    preds   = {}
    results = {}

    for model_name, model in models.items():
        print(f"  Fitting {model_name}...")
        try:
            if model_name == 'LightGBM':
                Xtr = pd.DataFrame(X_train_sc, columns=feature_names)
                Xte = pd.DataFrame(X_test_sc,  columns=feature_names)
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
            else:
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average='macro',     zero_division=0)
            f1w = f1_score(y_test, y_pred, average='weighted',  zero_division=0)
            prm = precision_score(y_test, y_pred, average='macro',    zero_division=0)
            rem = recall_score(y_test, y_pred, average='macro',       zero_division=0)
            prw = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rew = recall_score(y_test, y_pred, average='weighted',    zero_division=0)

            print(f"    Acc={acc:.4f}  F1-Macro={f1m:.4f}  F1-Weighted={f1w:.4f}")
            print(f"    {classification_report(y_test, y_pred, target_names=class_names, zero_division=0, digits=4)}")

            trained[model_name] = model
            preds[model_name]   = {'y_true': y_test, 'y_pred': y_pred}
            results[model_name] = {
                'accuracy_test':           acc,
                'f1_macro_test':           f1m,
                'f1_weighted_test':        f1w,
                'precision_macro_test':    prm,
                'recall_macro_test':       rem,
                'precision_weighted_test': prw,
                'recall_weighted_test':    rew,
                'classification_report':   classification_report(
                    y_test, y_pred, target_names=class_names,
                    zero_division=0, digits=4),
            }

        except Exception as e:
            print(f"    ERRO: {e}")
            import traceback; traceback.print_exc()
            results[model_name] = None

    return trained, preds, results

# ====================================================================
# CONFUSION MATRICES
# ====================================================================

def plot_confusion_matrices(preds, class_names):
    print(f"\nPlotting confusion matrices...")
    valid = {k: v for k, v in preds.items() if v is not None}
    n     = len(valid)
    cols  = min(n, 2); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))
    axes = np.array(axes).flatten()
    fig.suptitle('Confusion Matrices — Scenario 2 (144 features, seed=42)',
                 fontsize=14, fontweight='bold')

    for i, (mn, pd_) in enumerate(valid.items()):
        ax  = axes[i]
        cm  = confusion_matrix(pd_['y_true'], pd_['y_pred'])
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
        annot  = [[f"{cm[r,c]}\n({cm_pct[r,c]:.1f}%)"
                   for c in range(cm.shape[1])] for r in range(cm.shape[0])]
        sns.heatmap(cm, annot=annot, fmt='s', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        acc = accuracy_score(pd_['y_true'], pd_['y_pred'])
        f1m = f1_score(pd_['y_true'], pd_['y_pred'], average='macro', zero_division=0)
        ax.set_title(f'{mn}  Acc={acc:.3f}  F1={f1m:.3f}', fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    fp = os.path.join(plots_folder, 'confusion_matrices.png')
    plt.savefig(fp, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ confusion_matrices.png")

# ====================================================================
# SHAP ANALYSIS
# ====================================================================

def generate_shap_analysis(trained_models, X_train_sc, X_test_sc,
                           feature_names, class_names):
    print(f"\nSHAP Analysis...")
    X_train_df = pd.DataFrame(X_train_sc, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test_sc,  columns=feature_names)
    shap_rankings  = {}
    shap_per_class = {}   # {model_name: {class_name: {feature: score}}}

    for model_name, model in trained_models.items():
        print(f"\n  SHAP for {model_name}...")
        try:
            if isinstance(model, lgb.LGBMClassifier):
                print(f"    Usando TreeExplainer...")
                try:
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_df)
                except Exception as e:
                    print(f"    TreeExplainer falhou: {e} — fallback para KernelExplainer")
                    explainer   = shap.KernelExplainer(model.predict_proba, X_train_df)
                    shap_values = explainer.shap_values(X_test_df, nsamples='auto')
            else:
                print(f"    Usando KernelExplainer...")
                explainer   = shap.KernelExplainer(model.predict_proba, X_train_df)
                shap_values = explainer.shap_values(X_test_df, nsamples='auto')

            if shap_values is None:
                continue

            num_classes = len(class_names)
            if isinstance(shap_values, list) and len(shap_values) == num_classes:
                svp = shap_values
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 2 and num_classes == 2:
                    svp = [-shap_values, shap_values]
                elif shap_values.ndim == 3:
                    if shap_values.shape[2] == num_classes:
                        svp = [shap_values[:, :, i] for i in range(num_classes)]
                    elif shap_values.shape[0] == num_classes:
                        svp = [shap_values[i, :, :] for i in range(num_classes)]
                    else:
                        print(f"    Shape inesperado: {shap_values.shape}"); continue
                else:
                    print(f"    ndim inesperado: {shap_values.shape}"); continue
            else:
                print(f"    Formato não reconhecido: {type(shap_values)}"); continue

            mean_abs = [np.abs(svp[i]).mean(axis=0) for i in range(len(svp))]
            overall  = np.mean(mean_abs, axis=0)
            fi       = {feature_names[j]: overall[j] for j in range(len(overall))}
            sorted_f = sorted(fi.items(), key=lambda x: x[1], reverse=True)

            shap_rankings[model_name] = sorted_f
            print(f"    ✓ Top 5: {[f[0] for f in sorted_f[:5]]}")

            # ── Per-class summary plots ────────────────────────────────────
            for ci, cn in enumerate(class_names):
                if ci >= len(svp): continue
                plt.figure(figsize=(12, max(6, min(len(feature_names), 20) * 0.4)))
                try:
                    shap.summary_plot(svp[ci], X_test_df,
                                      feature_names=feature_names,
                                      show=False, max_display=20)
                    plt.title(f'SHAP Summary — {model_name} — Class: {cn}', fontsize=12)
                    fp = os.path.join(shap_folder,
                                      f'SHAP_Summary_{model_name}_Class_{cn}.png')
                    plt.savefig(fp, dpi=150, bbox_inches='tight')
                except Exception as e:
                    print(f"      Erro summary plot class {cn}: {e}")
                plt.close()

            # ── Per-class importance dict (mean |SHAP| per feature per class) ──
            # mean_abs[ci][j] = mean |SHAP| of feature j for class ci
            per_class_importance = {}   # {class_name: {feature: score}}
            for ci, cn in enumerate(class_names):
                if ci >= len(svp): continue
                ma = np.abs(svp[ci]).mean(axis=0)
                per_class_importance[cn] = {feature_names[j]: ma[j]
                                            for j in range(len(feature_names))}

            shap_per_class[model_name] = per_class_importance

            # ── Overall importance bar plot (mean across classes) ─────────────
            top20 = sorted_f[:20]
            plt.figure(figsize=(12, max(8, len(top20) * 0.4)))
            y_pos = np.arange(len(top20))
            plt.barh(y_pos, [v for _, v in top20], color='skyblue', alpha=0.8)
            plt.yticks(y_pos, [f for f, _ in top20])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'Feature Importance (SHAP) — {model_name} — Scenario 2\n(mean across all classes)')
            plt.grid(axis='x', alpha=0.3)
            plt.gca().invert_yaxis()
            mx = max(v for _, v in top20) if top20 else 1
            for i, (_, v) in enumerate(top20):
                plt.text(v + mx * 0.01, i, f'{v:.4f}', va='center', fontsize=9)
            plt.tight_layout()
            fp = os.path.join(shap_folder, f'SHAP_Importance_{model_name}.png')
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            plt.close()

            # ── Grouped bar plot: top-20 features × class ────────────────────
            top20_names   = [f for f, _ in top20]
            class_colors  = ['#4C72B0', '#DD8452', '#55A868']   # up to 3 classes
            n_feat        = len(top20_names)
            n_cls         = len(class_names)
            bar_width     = 0.8 / n_cls
            x             = np.arange(n_feat)

            fig_h = max(7, n_feat * 0.45)
            fig, ax = plt.subplots(figsize=(14, fig_h))

            for ci, cn in enumerate(class_names):
                if cn not in per_class_importance: continue
                vals   = [per_class_importance[cn].get(f, 0.0) for f in top20_names]
                offset = (ci - n_cls / 2 + 0.5) * bar_width
                bars   = ax.barh(x + offset, vals, bar_width,
                                 label=cn, color=class_colors[ci % len(class_colors)],
                                 alpha=0.85)

            ax.set_yticks(x)
            ax.set_yticklabels(top20_names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|', fontsize=11)
            ax.set_title(
                f'Feature Importance per Class (SHAP) — {model_name} — Scenario 2\n'
                f'Top-20 features by overall importance',
                fontsize=12, fontweight='bold')
            ax.legend(title='Class', fontsize=10, title_fontsize=10,
                      loc='lower right')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            fp = os.path.join(shap_folder, f'SHAP_Importance_PerClass_{model_name}.png')
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Plots salvos para {model_name}")

        except Exception as e:
            print(f"    SHAP error para {model_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Save rankings CSV (overall + per-class) ──────────────────────────
    rows = []
    for mn, ranking in shap_rankings.items():
        pci = shap_per_class.get(mn, {})
        for rank, (feat, score) in enumerate(ranking, 1):
            row = {'Model': mn, 'Rank': rank,
                   'Feature': feat, 'SHAP_Importance_Overall': score}
            for cn in class_names:
                row[f'SHAP_{cn}'] = pci.get(cn, {}).get(feat, float('nan'))
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_folder, 'shap_feature_rankings.csv'),
            index=False, float_format='%.12f')
        print(f"\n  ✓ shap_feature_rankings.csv salvo (com colunas por classe)")

    return shap_rankings

# ====================================================================
# LIGHTGBM NATIVE IMPORTANCE
# ====================================================================

def plot_lightgbm_importance(trained_models, feature_names):
    if 'LightGBM' not in trained_models:
        return
    model = trained_models['LightGBM']
    if not hasattr(model, 'booster_'):
        return
    print(f"\nLightGBM native importance plots...")
    for imp_type in ['gain', 'split']:
        native = model.booster_.feature_importance(importance_type=imp_type)
        pairs  = sorted(zip(feature_names, native), key=lambda x: x[1], reverse=True)
        top20  = pairs[:20]
        plt.figure(figsize=(12, max(8, len(top20) * 0.4)))
        y_pos = np.arange(len(top20))
        plt.barh(y_pos, [v for _, v in top20], color='lightcoral', alpha=0.8)
        plt.yticks(y_pos, [f for f, _ in top20])
        plt.xlabel(f'Importance ({imp_type})')
        plt.title(f'LightGBM Feature Importance ({imp_type}) — Scenario 2')
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()
        mx = max(v for _, v in top20) if top20 else 1
        for i, (_, v) in enumerate(top20):
            plt.text(v + mx * 0.01, i, f'{v:.1f}', va='center', fontsize=9)
        plt.tight_layout()
        fp = os.path.join(plots_folder, f'LightGBM_Importance_{imp_type}.png')
        plt.savefig(fp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ LightGBM_Importance_{imp_type}.png")

# ====================================================================
# MAIN
# ====================================================================

def main():
    print(f"\n{'='*70}")
    print(f" SCENARIO 2 — SHAP from Sc3 (Corr=None, Cum=1.00 | seed={RANDOM_SEED})")
    print(f" Hyperspace idêntico ao Scenario 3 | MLP_ARCH_MAP alinhado")
    print(f"{'='*70}")

    # 1. Carregar dados
    X_train, X_test, y_train, y_test, class_names = load_dataset()

    # 2. Carregar features selecionadas do combo Sc3 (todas as 144)
    features = load_selected_features()
    if features is None:
        print("ERRO: features não encontradas. Verifique COMBO_FOLDER.")
        return

    X_train_sel = X_train[features]
    X_test_sel  = X_test[features]

    # 3. Escalar
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_test_sc  = scaler.transform(X_test_sel)

    # 4. Reconstruir modelos a partir dos hiperparâmetros salvos pelo Sc3
    params_file = os.path.join(COMBO_FOLDER, 'best_hyperparameters.json')
    if not os.path.exists(params_file):
        print(f"ERRO: {params_file} não encontrado.")
        return
    models = rebuild_models(params_file)

    # 5. Fit + evaluate
    trained, preds, results = train_and_evaluate(
        models, X_train_sc, X_test_sc, y_train, y_test, class_names, features)

    # 6. Save results CSV
    rows = []
    for mn, res in results.items():
        if res is None:
            continue
        rows.append({
            'Model':                 mn,
            'Num_Features':          len(features),
            'Accuracy':              res['accuracy_test'],
            'F1_Macro':              res['f1_macro_test'],
            'F1_Weighted':           res['f1_weighted_test'],
            'Precision_Macro':       res['precision_macro_test'],
            'Recall_Macro':          res['recall_macro_test'],
            'Precision_Weighted':    res['precision_weighted_test'],
            'Recall_Weighted':       res['recall_weighted_test'],
            'Classification_Report': res['classification_report'],
        })
    csv_path = os.path.join(output_folder, 'model_results.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')
    print(f'\n\u2713 model_results.csv salvo em: {csv_path}')

    # 7. Confusion matrices
    plot_confusion_matrices(preds, class_names)

    # 8. SHAP
    shap_rankings = generate_shap_analysis(
        trained, X_train_sc, X_test_sc, features, class_names)

    # 9. LightGBM native importance
    plot_lightgbm_importance(trained, features)

    print(f"\n{'='*70}")
    print(f" CONCLUÍDO — resultados em: {output_folder}/")
    print(f" Seed usado: {RANDOM_SEED} (idêntico ao Scenario 3)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()