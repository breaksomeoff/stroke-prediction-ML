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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scripts import config  # Import delle variabili di configurazione

# Impostazione logging di base
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

#######################################################################
#                    Funzioni di Utilità per il Preprocessing
#######################################################################

def calculate_median_and_impute(df_train, column):
    """
    Calcola la mediana su df_train[column] e la usa per imputare i NaN.
    Ritorna la mediana per poterla applicare anche a validation/test.
    """
    median_value = df_train[column].median()
    df_train[column] = df_train[column].fillna(median_value)
    return median_value

def apply_impute(df, column, impute_value):
    """
    Usa la mediana calcolata sul train per imputare anche su validation/test.
    """
    df[column] = df[column].fillna(impute_value)

def create_label_encoder(train_series):
    """
    Crea e addestra un LabelEncoder su train_series (colonna del train).
    """
    le = LabelEncoder()
    le.fit(train_series.astype(str))
    return le

def apply_label_encoding(df_train, df_val, df_test, column):
    """
    Crea un LabelEncoder sul train e lo applica a train/val/test.
    Gestisce eventuali categorie sconosciute in val/test mappandole a -1.
    """
    le = create_label_encoder(df_train[column])
    # Encodiamo il train
    df_train[column] = le.transform(df_train[column].astype(str))

    # Encodiamo validation e test, gestendo eventuali categorie sconosciute
    for df_ in [df_val, df_test]:
        df_[column] = df_[column].apply(lambda x: x if x in le.classes_ else None)
        df_[column] = df_[column].astype("object").replace({None: "UNK"})
        if "UNK" not in le.classes_:
            new_classes = list(le.classes_) + ["UNK"]
            le.classes_ = np.array(new_classes, dtype=object)
        df_[column] = df_[column].map(lambda x: -1 if x == "UNK" else le.transform([x])[0])

#######################################################################
#                                Main
#######################################################################

def main():
    # 1) Caricamento dataset raw
    df = pd.read_csv(config.RAW_DATA_PATH)
    logging.info(f"[PRE-PROCESSING] Dataset raw caricato: {df.shape[0]} righe, {df.shape[1]} colonne.")

    # 2) Suddivisione in Train (70%), Validation (15%) e Test (15%)
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"[PRE-PROCESSING] Target '{config.TARGET_COLUMN}' non presente.")

    X_full = df.drop(columns=[config.TARGET_COLUMN])
    y_full = df[config.TARGET_COLUMN]

    # Primo split: Train 70% vs Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full,
        test_size=0.30,
        random_state=config.RANDOM_STATE,
        stratify=y_full
    )
    # Secondo split: Validation 15% e Test 15% (della totalità)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=config.RANDOM_STATE,
        stratify=y_temp
    )

    # Rimuoviamo le feature poco informative
    features_to_remove = ["id", "gender", "Residence_type"]
    X_train.drop(columns=features_to_remove, inplace=True, errors='ignore')
    X_val.drop(columns=features_to_remove, inplace=True, errors='ignore')
    X_test.drop(columns=features_to_remove, inplace=True, errors='ignore')

    # Riconcateniamo in DataFrame completi
    df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_val   = pd.concat([X_val,   y_val],   axis=1).reset_index(drop=True)
    df_test  = pd.concat([X_test,  y_test],  axis=1).reset_index(drop=True)

    logging.info(f"[PRE-PROCESSING] Train set: {X_train.shape[0]} righe | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # 3) Preprocessing (calcolato sul training set e applicato a validation/test)
    # 3.a Imputazione 'bmi'
    if "bmi" in df_train.columns:
        median_bmi = calculate_median_and_impute(df_train, "bmi")
        # Applica la mediana a validation e test
        apply_impute(df_val, "bmi", median_bmi)
        apply_impute(df_test, "bmi", median_bmi)
        logging.info("[PRE-PROCESSING] Imputazione per 'bmi' completata.")

    # 3.b Encoding variabili categoriche:
    #     Label Encoding per multi-level: 'work_type', 'smoking_status'
    cat_multi = ["work_type", "smoking_status"]
    for c in cat_multi:
        if c in df_train.columns:
            apply_label_encoding(df_train, df_val, df_test, c)
            logging.info(f"[PRE-PROCESSING] Encoding per '{c}' completato.")

    # Binarizzazione per 'ever_married'
    if "ever_married" in df_train.columns:
        bin_map = {"No": 0, "Yes": 1}
        df_train["ever_married"] = df_train["ever_married"].map(bin_map)
        df_val["ever_married"]   = df_val["ever_married"].map(bin_map)
        df_test["ever_married"]  = df_test["ever_married"].map(bin_map)
        logging.info("[PRE-PROCESSING] Binarizzazione per 'ever_married' completata.")

    # 4) Salvataggio dei dataset finali
    os.makedirs(os.path.dirname(config.TRAIN_DATA_PATH), exist_ok=True)

    df_train.to_csv(config.TRAIN_DATA_PATH, index=False)
    df_val.to_csv(config.VALIDATION_DATA_PATH, index=False)
    df_test.to_csv(config.TEST_DATA_PATH, index=False)

    logging.info(f"[PRE-PROCESSING] Train salvato: {df_train.shape[0]} righe")
    logging.info(f"[PRE-PROCESSING] Validation salvato: {df_val.shape[0]} righe")
    logging.info(f"[PRE-PROCESSING] Test salvato: {df_test.shape[0]} righe")

    # Distribuzione classi
    logging.info("[PRE-PROCESSING] Distribuzione TRAIN:")
    logging.info(f"\n{df_train[config.TARGET_COLUMN].value_counts()}")
    logging.info("[PRE-PROCESSING] Distribuzione VALIDATION:")
    logging.info(f"\n{df_val[config.TARGET_COLUMN].value_counts()}")
    logging.info("[PRE-PROCESSING] Distribuzione TEST:")
    logging.info(f"\n{df_test[config.TARGET_COLUMN].value_counts()}")
    logging.info("[SUCCESS] Preprocessing completato con successo.")

if __name__ == "__main__":
    main()