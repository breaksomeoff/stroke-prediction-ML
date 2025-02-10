# model.py
# Script per l'addestramento di un modello Random Forest finalizzato alla predizione dell'ictus.
# - Carica train.csv (non oversamplato) e validation.csv da data/processed/.
# - Esegue la Cross-Validation sul training set usando RandomizedSearchCV.
# - Ottimizza una soglia di classificazione sul validation set tramite la Precision-Recall Curve,
#   utilizzando il F2-score come metrica primaria.
# - Mostra metriche finali sul validation set e l'importanza delle feature.
# - Salva il modello e la soglia nei percorsi definiti in config.

import os
import logging
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, fbeta_score)
from joblib import dump
from scripts import config  # Import delle variabili di configurazione

# Impostazione logging di base
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def compute_cv_metrics(model, X, y, cv_splits=5):
    """
    Esegue una Stratified K-Fold cross-validation e calcola precision, recall, F2 e ROC-AUC per ogni fold.
    Restituisce i punteggi medi e stampa quelli di ogni fold.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    precision_list, recall_list, f2_list, roc_list = [], [], [], []
    fold_idx = 1
    for train_idx, test_idx in skf.split(X, y):
        X_fold_train, X_fold_test = X.iloc[train_idx], X.iloc[test_idx]
        y_fold_train, y_fold_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred_fold = model.predict(X_fold_test)
        prec = precision_score(y_fold_test, y_pred_fold)
        rec = recall_score(y_fold_test, y_pred_fold)
        f2 = fbeta_score(y_fold_test, y_pred_fold, beta=2)
        y_probs_fold = model.predict_proba(X_fold_test)[:, 1]
        roc = roc_auc_score(y_fold_test, y_probs_fold)

        logging.info(f"[MODEL] [CV Fold {fold_idx}] Precision: {prec:.4f} | Recall: {rec:.4f} | F2: {f2:.4f} | ROC-AUC: {roc:.4f}")
        precision_list.append(prec)
        recall_list.append(rec)
        f2_list.append(f2)
        roc_list.append(roc)
        fold_idx += 1

    return {
        "precision_mean": np.mean(precision_list),
        "recall_mean": np.mean(recall_list),
        "f2_mean": np.mean(f2_list),
        "roc_auc_mean": np.mean(roc_list)
    }

def find_optimal_threshold(model, X_val, y_val):
    """
    Usa la curva Precision-Recall sul validation set per trovare la soglia che massimizza il F2-score.
    Ritorna (best_threshold, best_f2).
    """
    y_probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
    beta = 2.0
    # Calcola F2-score per ciascuna soglia
    f2_scores = (1 + beta ** 2) * (precision * recall) / ((beta ** 2) * precision + recall + 1e-16)
    best_idx = np.argmax(f2_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f2 = f2_scores[best_idx]
    logging.info(f"[MODEL] [Threshold Optimization] Best F2: {best_f2:.4f} at threshold = {best_thresh:.3f}")
    return best_thresh, best_f2

def main():
    try:
        logging.info("[MODEL] Inizio addestramento modello Random Forest per stroke prediction.")

        # 1) Caricamento training e validation set
        train_path = config.TRAIN_DATA_PATH
        val_path   = config.VALIDATION_DATA_PATH

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file non trovato: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation file non trovato: {val_path}")

        df_train = pd.read_csv(train_path)
        df_val   = pd.read_csv(val_path)

        logging.info(f"[MODEL] Train set caricato: {df_train.shape[0]} righe, {df_train.shape[1]} colonne.")
        logging.info(f"[MODEL] Validation set caricato: {df_val.shape[0]} righe, {df_val.shape[1]} colonne.")

        X_train = df_train.drop(columns=[config.TARGET_COLUMN])
        y_train = df_train[config.TARGET_COLUMN]
        X_val = df_val.drop(columns=[config.TARGET_COLUMN])
        y_val = df_val[config.TARGET_COLUMN]

        # 2) Inizializzazione Random Forest base con class_weight='balanced'
        rf_base = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced')

        # 3) Ricerca iperparametri con RandomizedSearchCV + StratifiedKFold
        # Estendiamo lo spazio di ricerca per riflettere le scelte contestuali
        param_dist = {
            "n_estimators": [200, 300, 500, 800, 1000],
            "max_depth": [20, 30, 40, 50],
            "min_samples_split": [10, 20, 30, 40],
            "min_samples_leaf": [2, 5, 10],
            "max_features": [None, "sqrt", "log2", 0.5],
            "criterion": ["gini", "entropy"],
            "bootstrap": [True, False],
            "ccp_alpha": [0.0, 0.0001, 0.001]
        }

        # Creiamo un custom scorer per F2-score
        from sklearn.metrics import fbeta_score, make_scorer
        f2_scorer = make_scorer(fbeta_score, beta=2)

        scoring = {
            "f2": f2_scorer,
            "roc_auc": "roc_auc"
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

        random_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_dist,
            n_iter=100,
            scoring=scoring,
            refit="f2",  # Ottimizziamo il F2-score, dato che vogliamo privilegiare la recall
            cv=skf,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        logging.info("[MODEL] Inizio Ricerca Iperparametri RandomizedSearchCV...")
        random_search.fit(X_train, y_train)
        logging.info("[MODEL] Ricerca iperparametri completata.")

        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_
        logging.info(f"[MODEL] Migliori Parametri Trovati: {best_params}")

        mean_f2_cv = random_search.cv_results_["mean_test_f2"][random_search.best_index_]
        mean_roc_cv = random_search.cv_results_["mean_test_roc_auc"][random_search.best_index_]
        logging.info(f"[MODEL] Miglior F2 in CV: {mean_f2_cv:.4f} | Miglior ROC-AUC in CV: {mean_roc_cv:.4f}")

        # 4) Riaddestramento su TUTTO il training set con i migliori parametri
        best_rf.fit(X_train, y_train)

        logging.info("[MODEL] Valutazione con cross-validation su training set (solo a scopo diagnostico):")
        metrics_mean = compute_cv_metrics(best_rf, X_train, y_train, cv_splits=5)
        logging.info(f"[MODEL] [CV Summary] Precision: {metrics_mean['precision_mean']:.4f}, "
                     f"Recall: {metrics_mean['recall_mean']:.4f}, "
                     f"F2: {metrics_mean['f2_mean']:.4f}, "
                     f"ROC-AUC: {metrics_mean['roc_auc_mean']:.4f}")

        # 5) Ottimizzazione soglia sul validation set tramite il compromesso Precision-Recall (ottimizziamo F2)
        best_thresh, best_f2_val = find_optimal_threshold(best_rf, X_val, y_val)
        logging.info(f"[MODEL] Soglia ottimale (val) scelta = {best_thresh:.3f} (F2={best_f2_val:.4f})")

        # 6) Valutazione finale sul Validation Set usando la soglia di threshold ottimale
        y_probs_val = best_rf.predict_proba(X_val)[:, 1]
        y_pred_thresh = (y_probs_val >= best_thresh).astype(int)

        prec_val = precision_score(y_val, y_pred_thresh)
        rec_val  = recall_score(y_val, y_pred_thresh)
        f2_val   = fbeta_score(y_val, y_pred_thresh, beta=2)
        acc_val  = accuracy_score(y_val, y_pred_thresh)
        roc_val  = roc_auc_score(y_val, y_probs_val)

        logging.info(f"[MODEL] [VALIDATION] Accuracy={acc_val:.4f} | Precision={prec_val:.4f} | Recall={rec_val:.4f} "
                     f"| F2={f2_val:.4f} | ROC-AUC={roc_val:.4f}")

        # 7) Importanza delle feature
        feature_importances = best_rf.feature_importances_
        feature_names = X_train.columns
        imp_sorted_idx = np.argsort(feature_importances)[::-1]
        logging.info("[MODEL] [FEATURE IMPORTANCES - ordine decrescente]:")
        for idx in imp_sorted_idx:
            logging.info(f"  {feature_names[idx]}: {feature_importances[idx]:.4f}")

        # 8) Salvataggio del modello e della soglia ottimale
        os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
        dump(best_rf, config.MODEL_PATH)
        logging.info(f"[MODEL] Modello salvato in: {config.MODEL_PATH}")

        with open(config.THRESHOLD_OPTIMAL_PATH, "w") as f:
            f.write(str(best_thresh))
        logging.info(f"[MODEL] Soglia ottimale salvata in: {config.THRESHOLD_OPTIMAL_PATH}")

        logging.info("[SUCCESS] Addestramento completato con successo.")

    except Exception as e:
        logging.error(f"[FAIL] Errore durante l'addestramento del modello: {e}")
        raise e

if __name__ == "__main__":
    main()
