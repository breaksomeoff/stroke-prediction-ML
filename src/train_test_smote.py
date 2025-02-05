import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os


# --- Funzioni di supporto ---

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carica il dataset dal file CSV e restituisce un DataFrame.
    """
    return pd.read_csv(file_path)


def split_data(df: pd.DataFrame,
               target_col: str = "stroke",
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Suddivide il dataset in train e test, stratificando sul target.
    Non altera numericamente i dati, si limita a fare lo split.
    """
    x = df.drop(columns=[target_col])
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return x_train, x_test, y_train, y_test


def apply_smote(x_train: pd.DataFrame,
                y_train: pd.Series,
                random_state: int = 42) -> tuple:
    """
    Applica SMOTE sul train set per gestire la minoranza della classe (stroke=1).
    Restituisce (x_res, y_res) con le nuove istanze sintetiche create.
    """
    sm = SMOTE(random_state=random_state)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    return x_res, y_res


def snap_age_below_2(value: float) -> float:
    """
    Mappa un valore < 2 al multiplo di 0.08 più vicino,
    con un massimo di 2.0 (che poi formatteremo come intero se esattamente 2.0).
    Esempi:
      - 0.03 -> 0.08
      - 1.72 -> 1.72
      - 1.97 -> 2.0
    """
    snapped = round(value / 0.08) * 0.08
    if snapped < 0.08:
        snapped = 0.08
    if snapped >= 1.96:
        snapped = 2.0
    return snapped


def format_age(val: float) -> str:
    """
    - Se age >= 2: restituisce un intero (es. '35').
    - Se age < 2: approssima al multiplo di 0.08 più vicino.
      - Se il risultato è 2.0 -> '2'
      - Altrimenti lo formatta con 2 decimali, es. '1.72'.
    """
    if val >= 2:
        return str(int(round(val)))
    else:
        snapped = snap_age_below_2(val)
        return "2" if snapped == 2.0 else f"{snapped:.2f}"


def format_avg_glucose(val: float) -> str:
    """
    Arrotonda a 2 cifre decimali, poi rimuove gli zeri finali.
    Esempi:
      86.0 -> '86'
      60.4 -> '60.4'
      139.67 -> '139.67'
    """
    rounded = round(val, 2)
    return f"{rounded:.2f}".rstrip('0').rstrip('.')


def format_bmi(val: float) -> str:
    """
    Arrotonda a 1 cifra decimale, poi rimuove eventuali zeri finali.
    Esempi:
      19.0 -> '19'
      16.4 -> '16.4'
      23.45 -> '23.5'
    """
    rounded = round(val, 1)
    return f"{rounded:.1f}".rstrip('0').rstrip('.')


def format_columns_after_smote(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica la formattazione solo sulle colonne:
      - 'age' -> format_age
      - 'avg_glucose_level' -> format_avg_glucose
      - 'bmi' -> format_bmi
    Restituisce un nuovo DataFrame con valori formattati (come stringhe) in queste colonne.
    Le altre colonne rimangono intatte.
    """
    df_formatted = df.copy()
    if "age" in df_formatted.columns:
        df_formatted["age"] = df_formatted["age"].apply(format_age)
    if "avg_glucose_level" in df_formatted.columns:
        df_formatted["avg_glucose_level"] = df_formatted["avg_glucose_level"].apply(format_avg_glucose)
    if "bmi" in df_formatted.columns:
        df_formatted["bmi"] = df_formatted["bmi"].apply(format_bmi)
    return df_formatted


def train_test_smote(input_csv: str,
                     output_train_csv: str,
                     output_test_csv: str,
                     target_col: str = "stroke",
                     test_size: float = 0.2,
                     random_state: int = 42) -> None:
    """
    Pipeline:
      1) Carica il dataset pre-processato.
      2) Suddivide in train e test (stratificato).
      3) Applica SMOTE solo sul train.
      4) Applica la formattazione su alcune colonne (age, avg_glucose_level, bmi).
      5) Salva i due dataset (train e test) in file CSV.
    """
    # 1) Caricamento
    df = pd.read_csv(input_csv)

    # 2) Split train/test
    x_train, x_test, y_train, y_test = split_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )

    # 3) SMOTE sul solo train
    x_res, y_res = apply_smote(x_train, y_train, random_state=random_state)

    # 4) Formattazione delle colonne
    x_res_formatted = format_columns_after_smote(x_res)
    x_test_formatted = format_columns_after_smote(x_test)

    # Ricostruzione dei DataFrame
    train_df = pd.concat([x_res_formatted, y_res], axis=1)
    test_df = pd.concat([x_test_formatted, y_test], axis=1)

    # 5) Salvataggio dei dataset
    os.makedirs(os.path.dirname(output_train_csv), exist_ok=True)
    train_df.to_csv(output_train_csv, index=False)
    os.makedirs(os.path.dirname(output_test_csv), exist_ok=True)
    test_df.to_csv(output_test_csv, index=False)

    print(f"Train set (dopo SMOTE e formattazione) salvato in: {output_train_csv}")
    print(f"Test set salvato in: {output_test_csv}")


def process_and_split(rf_processed_path: str,
                      gb_processed_path: str,
                      rf_train_output: str,
                      rf_test_output: str,
                      gb_train_output: str,
                      gb_test_output: str,
                      target_col: str = "stroke",
                      test_size: float = 0.2,
                      random_state: int = 42):
    """
    Esegue lo split in train/test e SMOTE su due dataset pre-processati distinti:
      - Per Random Forest (dataset elaborato con label encoding)
      - Per Gradient Boosting (dataset elaborato con one-hot encoding)
    Salva i rispettivi train e test in file CSV separati.

    :param rf_processed_path: Path del dataset pre-processato per Random Forest.
    :param gb_processed_path: Path del dataset pre-processato per Gradient Boosting.
    :param rf_train_output: Path per salvare il train set per Random Forest.
    :param rf_test_output: Path per salvare il test set per Random Forest.
    :param gb_train_output: Path per salvare il train set per Gradient Boosting.
    :param gb_test_output: Path per salvare il test set per Gradient Boosting.
    :param target_col: Nome della colonna target (default: "stroke").
    :param test_size: Percentuale di dati da destinare al test (default: 0.2).
    :param random_state: Semino per la riproducibilità.
    """
    print("Processing split and SMOTE for Random Forest dataset (Label Encoding)...")
    train_test_smote(rf_processed_path, rf_train_output, rf_test_output,
                     target_col=target_col, test_size=test_size, random_state=random_state)

    print("Processing split and SMOTE for Gradient Boosting dataset (One-Hot Encoding)...")
    train_test_smote(gb_processed_path, gb_train_output, gb_test_output,
                     target_col=target_col, test_size=test_size, random_state=random_state)
