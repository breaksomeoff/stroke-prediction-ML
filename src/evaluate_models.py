"""
evaluate_model.py

Script per valutare e confrontare le performance di due modelli di classificazione:
- Random Forest (usando il test set pre-processato con label encoding)
- Gradient Boosting (usando il test set pre-processato con one-hot encoding)

Funzionalità:
1) Carica il dataset di test da CSV per ciascun modello (features + target='stroke').
2) Carica i due modelli salvati in formato .pkl.
3) Esegue la predizione sul test set per ciascun modello.
4) Calcola e stampa le metriche:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
5) Genera e visualizza:
   - Confusion Matrix per ogni modello (in due grafici affiancati)
   - ROC Curve comparativa (entrambi i modelli sullo stesso grafico)
6) Stampa un confronto in formato tabellare delle performance.

Esempio di utilizzo:
    evaluate_models(
        rf_test_data_path="../data/processed/test-stroke-data-rf.csv",
        gb_test_data_path="../data/processed/test-stroke-data-gb.csv",
        rf_model_path="../models/rf-stroke-model.pkl",
        gb_model_path="../models/gb-stroke-model.pkl"
    )
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)


def compute_metrics(y_true, y_pred, y_prob):
    """
    Calcola le metriche principali e le restituisce in un dizionario.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    }


def evaluate_model_single(test_data_path: str, model_path: str, model_name: str) -> dict:
    """
    Valuta un singolo modello sul test set, calcolando le metriche e visualizzando
    la Confusion Matrix.

    :param test_data_path: Percorso al CSV del dataset di test (con target 'stroke').
    :param model_path: Percorso al file .pkl del modello.
    :param model_name: Nome da usare per la stampa e per i grafici (es. "Random Forest").
    :return: Dizionario contenente le metriche calcolate.
    """
    # Caricamento del test set
    test_df = pd.read_csv(test_data_path)
    x_test = test_df.drop(columns=["stroke"])
    y_test = test_df["stroke"]

    # Caricamento del modello
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] {model_name} model loaded from: {model_path}")

    # Predizione
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    # Calcolo metriche
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Visualizzazione Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return metrics, y_test, y_proba  # restituisce anche y_test e y_proba per la ROC curve


def evaluate_models(rf_test_data_path: str,
                    gb_test_data_path: str,
                    rf_model_path: str,
                    gb_model_path: str):
    """
    Valuta e confronta le performance dei modelli Random Forest e Gradient Boosting sul test set.

    :param rf_test_data_path: Percorso del dataset di test per Random Forest.
    :param gb_test_data_path: Percorso del dataset di test per Gradient Boosting.
    :param rf_model_path: Percorso del modello Random Forest (.pkl).
    :param gb_model_path: Percorso del modello Gradient Boosting (.pkl).
    """
    print("\n=== Evaluating Random Forest Model ===")
    rf_metrics, y_test_rf, rf_proba = evaluate_model_single(rf_test_data_path, rf_model_path, "Random Forest")

    print("\n=== Evaluating Gradient Boosting Model ===")
    gb_metrics, y_test_gb, gb_proba = evaluate_model_single(gb_test_data_path, gb_model_path, "Gradient Boosting")

    # Verifichiamo che i target siano gli stessi in entrambi i test set
    # (idealmente dovrebbero esserlo, altrimenti i dataset non sono coerenti)
    if not y_test_rf.equals(y_test_gb):
        print("[WARNING] I target dei due test set non coincidono. Verifica la coerenza dei dati.")

    # Generazione ROC Curve comparativa
    plt.figure(figsize=(6, 5))
    # ROC per Random Forest
    ax = RocCurveDisplay.from_predictions(
        y_test_rf, rf_proba, name="Random Forest", color="blue"
    ).ax_
    # ROC per Gradient Boosting sullo stesso asse
    RocCurveDisplay.from_predictions(
        y_test_gb, gb_proba, name="Gradient Boosting", color="red", ax=ax
    )
    plt.title("ROC Curve Comparison")
    plt.show()

    # Confronto in formato tabellare
    comparison_df = pd.DataFrame({
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics
    })

    print("\n=== Comparison Table (Test Set) ===")
    print(comparison_df)
    print("\n[INFO] Evaluation completed. See confusion matrices, ROC curves, and metric table above.")
