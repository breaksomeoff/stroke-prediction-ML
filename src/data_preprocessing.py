# data_preprocessing.py
# Script migliorato per il preprocessing del dataset stroke-data.csv
# Obiettivo: creare Train (70%), Validation (15%) e Test (15%) set,
# applicando le trasformazioni (imputazione, encoding, ecc.) in modo da
# evitare data leakage.
#
# Passi principali:
#   1) Caricamento dataset
#   2) Creazione di train/val/test (70/15/15) con stratificazione su stroke
#   3) Preprocessing calcolato sul training set (imputazione, encoding, ecc.)
#   4) Applicazione delle stesse trasformazioni su validation/test
#   5) Salvataggio dei CSV finali (train, validation, test)
#
#   NB: Nessun oversampling/undersampling viene applicato (non verranno creati più dati artificiali in seguito ai tentativi precedenti).
#   Le trasformazioni (mediana, encoding) sono calcolate sul train set e poi applicate a val/test.

import os
import sys
import pandas as pd
import numpy as np
import logging
import joblib  # Import per salvare gli encoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scripts import config  # Import delle variabili di configurazione

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def create_label_encoder(train_series):
    """Crea e addestra un LabelEncoder su train_series."""
    le = LabelEncoder()
    le.fit(train_series.astype(str))
    return le

def apply_label_encoding(df_train, df_val, df_test, column, encoders_dict):
    """
    Crea un LabelEncoder per df_train[column], lo applica a train/val/test
    e lo salva in encoders_dict per uso futuro.
    """
    le = create_label_encoder(df_train[column])
    df_train[column] = le.transform(df_train[column].astype(str))

    for df_ in [df_val, df_test]:
        df_[column] = df_[column].apply(lambda x: x if x in le.classes_ else None)
        df_[column] = df_[column].astype("object").replace({None: "UNK"})
        if "UNK" not in le.classes_:
            le.classes_ = np.array(list(le.classes_) + ["UNK"], dtype=object)
        df_[column] = df_[column].map(lambda x: -1 if x == "UNK" else le.transform([x])[0])

    encoders_dict[column] = le  # Salva l'encoder

def main():
    df = pd.read_csv(config.RAW_DATA_PATH)
    logging.info(f"[PRE-PROCESSING] Dataset caricato: {df.shape}")

    X_full = df.drop(columns=[config.TARGET_COLUMN])
    y_full = df[config.TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full, test_size=0.30, random_state=config.RANDOM_STATE, stratify=y_full
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.RANDOM_STATE, stratify=y_temp
    )

    features_to_remove = ["id", "gender", "Residence_type"]
    X_train = X_train.drop(columns=features_to_remove, errors='ignore')
    X_val = X_val.drop(columns=features_to_remove, errors='ignore')
    X_test = X_test.drop(columns=features_to_remove, errors='ignore')

    df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_val   = pd.concat([X_val, y_val], axis=1).reset_index(drop=True)
    df_test  = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    logging.info(f"[PRE-PROCESSING] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    if "bmi" in df_train.columns:
        median_bmi = df_train["bmi"].median()
        df_train["bmi"] = df_train["bmi"].fillna(median_bmi)
        df_val["bmi"] = df_val["bmi"].fillna(median_bmi)
        df_test["bmi"] = df_test["bmi"].fillna(median_bmi)
        logging.info("[PRE-PROCESSING] Imputazione per 'bmi' completata.")

    encoders = {}
    cat_multi = ["work_type", "smoking_status"]
    for c in cat_multi:
        if c in df_train.columns:
            apply_label_encoding(df_train, df_val, df_test, c, encoders)
            logging.info(f"[PRE-PROCESSING] Encoding per '{c}' completato.")

    if "ever_married" in df_train.columns:
        bin_map = {"No": 0, "Yes": 1}
        df_train["ever_married"] = df_train["ever_married"].map(bin_map)
        df_val["ever_married"]   = df_val["ever_married"].map(bin_map)
        df_test["ever_married"]  = df_test["ever_married"].map(bin_map)
        logging.info("[PRE-PROCESSING] Binarizzazione per 'ever_married' completata.")

    os.makedirs(os.path.dirname(config.TRAIN_DATA_PATH), exist_ok=True)
    df_train.to_csv(config.TRAIN_DATA_PATH, index=False)
    df_val.to_csv(config.VALIDATION_DATA_PATH, index=False)
    df_test.to_csv(config.TEST_DATA_PATH, index=False)

    joblib.dump(encoders, os.path.join(os.path.dirname(config.ENCODER_PATH), "label_encoders.joblib"))
    logging.info(f"[PRE-PROCESSING] Label Encoders salvati con successo in \"{config.ENCODER_PATH}\"")

if __name__ == "__main__":
    main()