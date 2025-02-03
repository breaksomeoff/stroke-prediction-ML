import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carica il dataset dal file CSV e restituisce un DataFrame.
    """
    return pd.read_csv(file_path)


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


def encode_categorical(df: pd.DataFrame, encoding_type: str = "one-hot") -> pd.DataFrame:
    """
    Esegue l'encoding delle variabili categoriche.
    - 'one-hot' per Random Forest
    - 'label' per XGBoost
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    if encoding_type == "one-hot":
        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    elif encoding_type == "label":
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        df_encoded = df
    else:
        raise ValueError("encoding_type deve essere 'one-hot' o 'label'")

    return df_encoded


def remove_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rimuove le colonne One-Hot complementari non desiderate,
    mantenendo una sola colonna per variabile binaria.
    """
    columns_to_remove = [
        "gender_Female",
        "ever_married_No",
        "Residence_type_Rural"
    ]
    for col in columns_to_remove:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def preprocess_data(file_path: str, output_path: str, encoding_type: str = "one-hot") -> pd.DataFrame:
    """
    Esegue l'intero flusso di pre-processing:
      1) Caricamento dati
      2) Merge categorie rare ('Never_worked' + 'children' -> 'No_job/Children')
      3) Imputazione 'bmi' con mediana
      4) Encoding delle categoriche
      5) Rimozione delle colonne complementari per le variabili binarie
    """
    # 1) Caricamento dati
    df = load_data(file_path)

    # 2) Merge delle categorie rare
    df = merge_rare_categories(df)

    # 3) Imputazione dei valori mancanti in 'bmi'
    df = impute_bmi_with_median(df)

    # Rimuoviamo la colonna 'id' se presente
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # 4) Encoding delle variabili categoriche
    df = encode_categorical(df, encoding_type)

    # 5) Rimozione colonne indesiderate
    df = remove_unwanted_columns(df)

    # Salvataggio del dataset preprocessato
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset pre-processato salvato in: {output_path}")

    return df
