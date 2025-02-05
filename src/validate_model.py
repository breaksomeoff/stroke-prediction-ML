"""
validate_model.py

Script per eseguire K-Fold Cross Validation
su un modello pre-addestrato (Random Forest o Gradient Boosting), caricato da un file .pkl.

Funzionalità:
1) Carica il dataset di training da CSV.
2) Carica il modello salvato (pre-addestrato) da un file .pkl.
3) Applica K-Fold Cross Validation sullo stesso train set.
4) Calcola e stampa per ogni fold:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC
5) Genera un Boxplot per visualizzare la distribuzione di tali metriche sui K folds.
6) Restituisce (in stampa) la media e la deviazione standard per ciascuna metrica.

La funzione `validate_model()` viene chiamata all'interno di `validate_models()`
per eseguire la validazione separata su:
    - Il modello Random Forest (usando RANDOM_FOREST_TRAIN_DATA_PATH)
    - Il modello Gradient Boosting (usando GRADIENT_BOOSTING_TRAIN_DATA_PATH)

Esempio di utilizzo:
    validate_models(
        rf_model_path="path/al/rf-model.pkl",
        gb_model_path="path/al/gb-model.pkl",
        rf_train_data_path="path/al/train-rf.csv",
        gb_train_data_path="path/al/train-gb.csv",
        k=10
    )
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold

# Utilizziamo le stringhe standard per lo scoring, così scikit-learn utilizza i propri scorer interni
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}


def validate_model(model_path: str,
                   train_data_path: str,
                   k: int):
    """
    Esegue una K-Fold Cross Validation sul modello pre-addestrato specificato,
    calcolando e visualizzando le metriche di performance.

    :param model_path: Percorso del file .pkl contenente il modello (Random Forest o Gradient Boosting).
    :param train_data_path: Percorso del file CSV con il dataset di training (features + target='stroke').
    :param k: Numero di folds per la cross validation.
    """
    # 1) Caricamento del dataset di training
    train_df = pd.read_csv(train_data_path)
    x = train_df.drop(columns=["stroke"])
    y = train_df["stroke"]

    # 2) Caricamento del modello pre-addestrato
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded from: {model_path}")

    # 3) Definizione K-Fold (usiamo StratifiedKFold per mantenere la proporzione delle classi)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # 4) Esecuzione della cross-validation
    cv_results = cross_validate(
        estimator=model,
        X=x,
        y=y,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )

    # Estrazione dei risultati (le metriche sono sotto chiavi "test_<metric>")
    accuracy_scores = cv_results['test_accuracy']
    precision_scores = cv_results['test_precision']
    recall_scores = cv_results['test_recall']
    f1_scores = cv_results['test_f1']
    roc_scores = cv_results['test_roc_auc']

    # 5) Stampa dei risultati fold-by-fold
    print("\n=== K-FOLD CROSS VALIDATION RESULTS (Fold by Fold) ===")
    for i in range(k):
        print(f"\nFold {i + 1}")
        print(f" Accuracy:  {accuracy_scores[i]:.4f}")
        print(f" Precision: {precision_scores[i]:.4f}")
        print(f" Recall:    {recall_scores[i]:.4f}")
        print(f" F1-score:  {f1_scores[i]:.4f}")
        print(f" ROC-AUC:   {roc_scores[i]:.4f}")

    # 6) Creazione di un DataFrame per analisi e boxplot
    metrics_df = pd.DataFrame({
        'Accuracy': accuracy_scores,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-score': f1_scores,
        'ROC-AUC': roc_scores
    })

    # Generazione del boxplot delle metriche
    plt.figure(figsize=(8, 5))
    metrics_df.boxplot(grid=False)
    plt.title('K-Fold Cross Validation Metrics Distribution')
    plt.ylabel('Score')
    plt.show()

    # 7) Calcolo e stampa della media e della deviazione standard
    print("\n=== MEAN AND STANDARD DEVIATION ACROSS FOLDS ===")
    means = metrics_df.mean()
    stds = metrics_df.std()
    for metric in metrics_df.columns:
        print(f"{metric}: Mean = {means[metric]:.4f}, Std = {stds[metric]:.4f}")


def validate_models(rf_model_path: str,
                    gb_model_path: str,
                    rf_train_data_path: str,
                    gb_train_data_path: str,
                    k: int):
    """
    Esegue la validazione in cross-validation per entrambi i modelli
    (Random Forest e Gradient Boosting) utilizzando i rispettivi dataset di training.

    :param rf_model_path: Percorso del modello Random Forest (.pkl).
    :param gb_model_path: Percorso del modello Gradient Boosting (.pkl).
    :param rf_train_data_path: Percorso del dataset di training per Random Forest.
    :param gb_train_data_path: Percorso del dataset di training per Gradient Boosting.
    :param k: Numero di folds per la cross validation.
    """
    print("=== Validating Random Forest Model ===")
    validate_model(model_path=rf_model_path, train_data_path=rf_train_data_path, k=k)

    print("\n=== Validating Gradient Boosting Model ===")
    validate_model(model_path=gb_model_path, train_data_path=gb_train_data_path, k=k)
