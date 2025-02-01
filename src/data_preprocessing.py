import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carica il dataset dal file CSV e restituisce un DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def merge_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unisce le categorie 'Never_worked' e 'children' della colonna 'work_type'
    in un'unica categoria 'No_job/Children', se presenti.
    """
    if 'work_type' in df.columns:
        df['work_type'] = df['work_type'].replace(
            {
                'Never_worked': 'No_job/Children',
                'children': 'No_job/Children'
            }
        )
    return df

def impute_bmi_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa i valori mancanti di 'bmi' con la mediana.
    """
    if 'bmi' in df.columns:
        median_bmi = df['bmi'].median(skipna=True)
        df['bmi'] = df['bmi'].fillna(median_bmi)
    return df


def encode_categorical(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Esegue l'encoding delle variabili categoriche.
    Di default usa drop_first=True per evitare collinearità.
    """
    # Identifichiamo le colonne categoriche/oggetto
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=drop_first)
    return df_encoded


def preprocess_data(file_path: str, output_path: str) -> pd.DataFrame:
    """
    Esegue l'intero flusso di pre-processing base:
      1) Caricamento dati
      2) Merge categorie rare ('Never_worked' -> 'children')
      3) Imputazione 'bmi' con mediana
      4) One-Hot Encoding per tutte le categoriche

    Restituisce un DataFrame pronto per lo split in train e test.
    """
    # Carica il dataset
    df = load_data(file_path)

    # Unione categorie rare ('Never_worked' e 'children' -> 'No_job/Children'
    df = merge_rare_categories(df)

    # Imputazione mediana BMI
    df = impute_bmi_with_median(df)

    df.drop(columns=['id'], inplace=True)

    # Encoding delle categoriche
    df = encode_categorical(df, drop_first=True)

    # Crea la cartella 'data/processed' se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Salvare il DataFrame pre-processato
    df.to_csv(output_path, index=False)

    print(f"Dataset pre-processato salvato in: {output_path}")

    return df