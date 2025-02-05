"""
hyperparameter_tuning.py

Questo modulo esegue l'ottimizzazione degli iperparametri per:
 - Random Forest (usando il train set pre-processato con label encoding)
 - Gradient Boosting (usando il train set pre-processato con one-hot encoding)

Utilizza RandomizedSearchCV per cercare in modo randomizzato il miglior set di iperparametri,
e salva i modelli ottimizzati in formato pickle (.pkl).

Esempio di utilizzo:
    tune_models(
        rf_train_data_path="../data/processed/train-stroke-data-rf.csv",
        gb_train_data_path="../data/processed/train-stroke-data-gb.csv",
        rf_model_path="../models/rf-stroke-model_tuned.pkl",
        gb_model_path="../models/gb-stroke-model_tuned.pkl",
        n_iter=50,
        random_state=42
    )
"""

import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def tune_random_forest(train_data_path: str, random_state: int = 42, n_iter: int = 50):
    """
    Esegue il tuning degli iperparametri per il modello Random Forest utilizzando RandomizedSearchCV.

    :param train_data_path: Percorso del train set pre-processato per Random Forest.
    :param random_state: Seme per la riproducibilità.
    :param n_iter: Numero di combinazioni da provare.
    :return: Il miglior modello Random Forest trovato.
    """
    # Carica il dataset di training
    df = pd.read_csv(train_data_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # Inizializza il modello di base
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)

    # Definizione dello spazio degli iperparametri
    param_dist = {
        'n_estimators': [100, 250, 500, 750],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    # Inizializza RandomizedSearchCV
    rs_rf = RandomizedSearchCV(estimator=rf,
                               param_distributions=param_dist,
                               n_iter=n_iter,
                               cv=5,
                               scoring='f1',
                               random_state=random_state,
                               n_jobs=-1)
    rs_rf.fit(X, y)
    print("[INFO] Best parameters for Random Forest:", rs_rf.best_params_)
    return rs_rf.best_estimator_


def tune_gradient_boosting(train_data_path: str, random_state: int = 42, n_iter: int = 50):
    """
    Esegue il tuning degli iperparametri per il modello Gradient Boosting utilizzando RandomizedSearchCV.

    :param train_data_path: Percorso del train set pre-processato per Gradient Boosting.
    :param random_state: Seme per la riproducibilità.
    :param n_iter: Numero di combinazioni da provare.
    :return: Il miglior modello Gradient Boosting trovato.
    """
    # Carica il dataset di training
    df = pd.read_csv(train_data_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    # Inizializza il modello di base
    gb = GradientBoostingClassifier(random_state=random_state)

    # Definizione dello spazio degli iperparametri
    param_dist = {
        'n_estimators': [100, 250, 500, 750],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'subsample': [0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    # Inizializza RandomizedSearchCV
    rs_gb = RandomizedSearchCV(estimator=gb,
                               param_distributions=param_dist,
                               n_iter=n_iter,
                               cv=5,
                               scoring='f1',
                               random_state=random_state,
                               n_jobs=-1)
    rs_gb.fit(X, y)
    print("[INFO] Best parameters for Gradient Boosting:", rs_gb.best_params_)
    return rs_gb.best_estimator_


def tune_models(rf_train_data_path: str,
                gb_train_data_path: str,
                rf_model_path: str,
                gb_model_path: str,
                random_state: int = 42,
                n_iter: int = 50):
    """
    Allena (tuning) entrambi i modelli (Random Forest e Gradient Boosting)
    utilizzando RandomizedSearchCV e salva i modelli ottimizzati in file separati.

    :param rf_train_data_path: Percorso del train set per Random Forest.
    :param gb_train_data_path: Percorso del train set per Gradient Boosting.
    :param rf_model_path: Percorso in cui salvare il modello Random Forest ottimizzato.
    :param gb_model_path: Percorso in cui salvare il modello Gradient Boosting ottimizzato.
    :param random_state: Seme per la riproducibilità.
    :param n_iter: Numero di combinazioni da testare in RandomizedSearchCV.
    """
    print("=== Tuning Random Forest Model ===")
    best_rf = tune_random_forest(rf_train_data_path, random_state=random_state, n_iter=n_iter)

    print("\n=== Tuning Gradient Boosting Model ===")
    best_gb = tune_gradient_boosting(gb_train_data_path, random_state=random_state, n_iter=n_iter)

    # Salva i modelli ottimizzati
    with open(rf_model_path, "wb") as f:
        pickle.dump(best_rf, f)
    with open(gb_model_path, "wb") as f:
        pickle.dump(best_gb, f)

    print(f"\n[Tuned Model] Random Forest saved to: {rf_model_path}")
    print(f"[Tuned Model] Gradient Boosting saved to: {gb_model_path}")
