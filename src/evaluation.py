# evaluation.py
# Script per la valutazione finale del modello Random Forest per la predizione dell'ictus.
# Il modello viene caricato (salvato in MODEL_PATH) e testato sul test set (TEST_DATA_PATH).
# Vengono calcolate le metriche: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
# La soglia di classificazione viene letta dal file specificato in THRESHOLD_OPTIMAL_PATH.
# Vengono generati grafici: Matrice di Confusione, ROC Curve, Importanza delle Feature,
# e Precision-Recall Curve.
# Un report finale con le metriche e i grafici viene generato e salvato.

import os
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve)
from joblib import load
from scripts import config  # Import delle variabili di configurazione

# Impostazione logging di base
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def plot_confusion_matrix(cm, classes, title="Matrice di Confusione", cmap=None, save_path=None):
    # Plotta la matrice di confusione.
    if cmap is None:
        cmap = plt.get_cmap("Blues")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        logging.info(f"[EVALUATION] Matrice di confusione salvata in: {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path=None):
    """
    Calcola e plotta la ROC Curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
        logging.info(f"[EVALUATION] ROC Curve salvata in: {save_path}")
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    """
    Calcola e plotta la Precision-Recall Curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="darkgreen", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    if save_path:
        plt.savefig(save_path)
        logging.info(f"[EVALUATION] Precision-Recall Curve salvata in: {save_path}")
    plt.close()

def plot_feature_importances(feature_names, importances, save_path=None):
    """
    Plotta un grafico a barre delle feature importances.
    """
    idx_sorted = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[idx_sorted], y=np.array(feature_names)[idx_sorted])
    plt.xlabel("Importanza")
    plt.ylabel("Feature")
    plt.title("Importanza delle Feature")
    if save_path:
        plt.savefig(save_path)
        logging.info(f"[EVALUATION] Grafico importanza feature salvato in: {save_path}")
    plt.close()

def generate_report(metrics_dict, report_path):
    """
    Genera un report testuale con le metriche finali.
    """
    with open(report_path, "w") as f:
        f.write("===== REPORT FINALE DI VALUTAZIONE =====\n\n")
        for key, value in metrics_dict.items():
            f.write(f"{key}: {value:.4f}\n")
    logging.info(f"[EVALUATION] Report finale salvato in: {report_path}")

def main():
    try:
        logging.info("[EVALUATION] Inizio valutazione finale del modello Random Forest per stroke prediction.")

        # 1) Caricamento del modello finale e del test set
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(f"[EVALUATION] Modello non trovato: {config.MODEL_PATH}")
        model = load(config.MODEL_PATH)
        logging.info(f"[EVALUATION] Modello caricato da: {config.MODEL_PATH}")

        if not os.path.exists(config.TEST_DATA_PATH):
            raise FileNotFoundError(f"[EVALUATION] Test set non trovato: {config.TEST_DATA_PATH}")
        df_test = pd.read_csv(config.TEST_DATA_PATH)
        logging.info(f"[EVALUATION] Test set caricato: {df_test.shape[0]} righe, {df_test.shape[1]} colonne.")

        # Separiamo feature e target
        X_test = df_test.drop(columns=[config.TARGET_COLUMN])
        y_test = df_test[config.TARGET_COLUMN]

        # 2) Predizioni sul test set
        y_probs = model.predict_proba(X_test)[:, 1]

        # 3) Caricamento della soglia ottimale dal file
        if os.path.exists(config.THRESHOLD_OPTIMAL_PATH):
            with open(config.THRESHOLD_OPTIMAL_PATH, "r") as f:
                best_thresh = float(f.read().strip())
            logging.info(f"[EVALUATION] Soglia ottimale caricata da file: {best_thresh:.3f}")
        else:
            raise FileNotFoundError(f"[EVALUATION] File della soglia non trovato: {config.THRESHOLD_OPTIMAL_PATH}")

        # 4) Calcolo delle predizioni finali usando la soglia
        y_pred = (y_probs >= best_thresh).astype(int)

        # 5) Calcolo delle metriche
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_probs)

        logging.info(f"[EVALUATION] [TEST] Accuracy={accuracy:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | F1={f1:.4f} | ROC-AUC={roc_auc:.4f}")

        # 6) Visualizzazioni: salviamo tutti i plot in MODEL_PLOTS_PATH
        os.makedirs(config.MODEL_PLOTS_PATH, exist_ok=True)

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(config.MODEL_PLOTS_PATH, "confusion_matrix.png")
        plot_confusion_matrix(cm, classes=["0", "1"], title="Matrice di Confusione", save_path=cm_path)

        # ROC Curve
        roc_curve_path = os.path.join(config.MODEL_PLOTS_PATH, "roc_curve.png")
        plot_roc_curve(y_test, y_probs, save_path=roc_curve_path)

        # Precision-Recall Curve
        pr_curve_path = os.path.join(config.MODEL_PLOTS_PATH, "precision_recall_curve.png")
        plot_precision_recall_curve(y_test, y_probs, save_path=pr_curve_path)

        # Importanza delle feature
        feat_imp_path = os.path.join(config.MODEL_PLOTS_PATH, "feature_importances.png")
        plot_feature_importances(X_test.columns.tolist(), model.feature_importances_, save_path=feat_imp_path)

        # 7) Genera un report finale
        report_metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
            "Soglia Ottimale": best_thresh
        }
        generate_report(report_metrics, config.MODEL_REPORT_PATH)

        logging.info("[SUCCESS] Fine valutazione del modello Random Forest per stroke prediction.")

    except Exception as e:
        logging.error(f"[FAIL] Errore nella valutazione del modello: {e}")
        raise e

if __name__ == "__main__":
    main()
