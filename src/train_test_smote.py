import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def split_data(df: pd.DataFrame,
               target_col: str = "stroke",
               test_size: float = 0.2,
               random_state: int = 42):
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
        stratify=y  # mantiene le proporzioni di stroke=0/1 in train e test
    )
    return x_train, x_test, y_train, y_test


def apply_smote(x_train: pd.DataFrame,
                y_train: pd.Series,
                random_state: int = 42):
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
    # Arrotondo al multiplo di 0.08 più vicino
    snapped = round(value / 0.08) * 0.08

    # Forziamo un minimo di 0.08 (se volessimo evitare età < 0.08)
    if snapped < 0.08:
        snapped = 0.08

    # Se supera ~1.96, portalo a 2.0
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
        # Intero
        return str(int(round(val)))  # '35'
    else:
        snapped = snap_age_below_2(val)
        if snapped == 2.0:
            return "2"
        else:
            # due decimali
            return f"{snapped:.2f}"


def format_avg_glucose(val: float) -> str:
    """
    Arrotonda a 2 cifre decimali, poi rimuove gli zeri finali.
    Esempi:
      86.0 -> '86'
      60.4 -> '60.4'
      139.67 -> '139.67'
      57.09 -> '57.09'
    """
    rounded = round(val, 2)
    s = f"{rounded:.2f}".rstrip('0').rstrip('.')
    return s


def format_bmi(val: float) -> str:
    """
    Arrotonda a 1 cifra decimale, poi rimuove eventuali zeri finali.
    Esempi:
      19.0 -> '19'
      16.4 -> '16.4'
      23.45 -> '23.5'
    """
    rounded = round(val, 1)
    s = f"{rounded:.1f}".rstrip('0').rstrip('.')
    return s


def format_columns_after_smote(x: pd.DataFrame) -> pd.DataFrame:
    """
    Applica la formattazione *solo* sulle colonne:
      - 'age' -> format_age
      - 'avg_glucose_level' -> format_avg_glucose
      - 'bmi' -> format_bmi
    Restituisce un nuovo DataFrame con valori stringa in queste colonne.
    Le altre colonne rimangono intatte.
    """
    x_formatted = x.copy()

    if "age" in x_formatted.columns:
        x_formatted["age"] = x_formatted["age"].apply(format_age)

    if "avg_glucose_level" in x_formatted.columns:
        x_formatted["avg_glucose_level"] = x_formatted["avg_glucose_level"].apply(format_avg_glucose)

    if "bmi" in x_formatted.columns:
        x_formatted["bmi"] = x_formatted["bmi"].apply(format_bmi)

    return x_formatted


def train_test_smote(input_csv: str,
         output_train_csv: str,
         output_test_csv: str,
         target_col: str = "stroke",
         test_size: float = 0.2,
         random_state: int = 42):
    """
    Pipeline:
      1) Carica il dataset pre-processato (già con One-Hot, etc.)
      2) Split train/test (stratificato)
      3) SMOTE sul train
      4) Applica la formattazione 'ad hoc' su age, avg_glucose_level, bmi nel train SMOTEd
      5) (Opzionale) formatta anche il test, se lo vuoi omogeneo
      6) Salva i due CSV
    """
    # 1. Caricamento
    df = pd.read_csv(input_csv)

    # 2. Split train/test
    x_train, x_test, y_train, y_test = split_data(
        df,
        target_col=target_col,
        test_size=test_size,
        random_state=random_state
    )

    # 3. SMOTE solo sul train
    x_res, y_res = apply_smote(x_train, y_train, random_state=random_state)

    # 4. Formattazione specializzata su (x_res)
    x_res_formatted = format_columns_after_smote(x_res)

    # Formattiamo anche il test per assicurare coerenza con i formati originali dei dati
    x_test_formatted = format_columns_after_smote(x_test)

    # Ricostruisco i DataFrame con la colonna target
    train_df = pd.concat([x_res_formatted, y_res], axis=1)
    test_df = pd.concat([x_test_formatted, y_test], axis=1)

    # 5. Salvataggio su CSV
    train_df.to_csv(output_train_csv, index=False)
    test_df.to_csv(output_test_csv, index=False)

    print(f"Train set (dopo SMOTE e formattazione) salvato in: {output_train_csv}")
    print(f"Test set salvato in: {output_test_csv}")
