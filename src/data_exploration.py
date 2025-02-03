import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway

"""
EDA Script for the Stroke Prediction Dataset
Dataset: stroke-data.csv (located in data/raw/)
Questo script esegue l'analisi esplorativa dei dati (EDA).
Utilizza Pandas, Seaborn, Matplotlib, SciPy e Statsmodels per:
    - Caricamento e overview del dataset
    - Analisi dei valori mancanti e duplicati
    - Visualizzazione delle distribuzioni delle variabili numeriche e categoriche
    - Analisi delle correlazioni (numeriche e categoriche) tramite:
        * Heatmap (correlazione di Pearson)
        * One-Hot Encoding
        * Cramér’s V per variabili categoriche
        * ANOVA per testare l’effetto delle categorie sulle variabili numeriche
        * Test del chi-quadro per l'associazione tra variabili categoriche e il target stroke
    - Estrazione di insight preliminari e suggerimenti di pre-processing
"""

# Configurazione globale per i plot
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_theme(style='whitegrid')


# =============================================================================
# FUNZIONI DI SUPPORTO
# =============================================================================

def load_dataset(file_path):
    """
    Carica il dataset da un file CSV.
    :param file_path: Percorso del file CSV.
    :return: DataFrame Pandas contenente il dataset.
    """
    df = pd.read_csv(file_path)
    return df


def overview_data(df):
    """
    Mostra una panoramica del dataset: prime 5 righe, info e statistiche descrittive.
    """
    print("=== Prime 5 righe del dataset ===")
    print(df.head(), "\n")

    print("=== Informazioni sul dataset ===")
    df.info()
    print()

    print("=== Statistiche descrittive delle variabili numeriche ===")
    print(df.describe(), "\n")


def analyze_missing_duplicates(df):
    """
    Analizza i valori mancanti e i duplicati nel dataset.
    """
    print("=== Valori mancanti per colonna ===")
    missing_counts = df.isnull().sum()
    print(missing_counts, "\n")

    print("=== Percentuale di valori mancanti per colonna (%) ===")
    missing_percent = (df.isnull().mean() * 100).round(2)
    print(missing_percent, "\n")

    print("=== Numero di record duplicati ===")
    duplicates = df.duplicated().sum()
    print(duplicates, "\n")


def plot_numeric_distributions(df, num_features):
    """
    Genera istogrammi con KDE per le variabili numeriche.
    :param df: DataFrame contenente i dati.
    :param num_features: Lista di nomi delle colonne numeriche.
    """
    for feature in num_features:
        plt.figure()
        sns.histplot(df[feature].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f'Distribuzione di {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequenza')
        plt.tight_layout()
        plt.show()


def plot_categorical_distributions(df, cat_features):
    """
    Genera barplot ordinati per frequenza per le variabili categoriche.
    :param df: DataFrame contenente i dati.
    :param cat_features: Lista di nomi delle colonne categoriche.
    """
    for feature in cat_features:
        plt.figure()
        # Ordinamento per frequenza decrescente
        order = df[feature].value_counts().index
        sns.countplot(
            x=feature, data=df, order=order,
            hue=feature,
            dodge=False,
            palette='viridis',
            legend=False
        )
        plt.title(f'Conteggio per la variabile {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequenza')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Stampa la distribuzione testuale
        print(f"=== Distribuzione della variabile '{feature}' ===")
        print(df[feature].value_counts(), "\n")


def plot_target_distribution(df, target):
    """
    Visualizza la distribuzione della classe target tramite un countplot.
    :param df: DataFrame contenente i dati.
    :param target: Nome della variabile target.
    """
    plt.figure()
    sns.countplot(x=target, data=df, hue=target, dodge=False, palette='Set2', legend=False)
    plt.title(f"Distribuzione della classe target '{target}'")
    plt.xlabel(target)
    plt.ylabel('Conteggio')
    plt.tight_layout()
    plt.show()

    print(f"=== Distribuzione della variabile target '{target}' ===")
    print(df[target].value_counts(), "\n")


def plot_numeric_correlation_heatmap(df):
    """
    Calcola e visualizza la heatmap della matrice di correlazione per le variabili numeriche.
    """
    # Seleziona solo le colonne numeriche
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    print("=== Matrice di correlazione (variabili numeriche) ===")
    print(corr, "\n")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap delle correlazioni (variabili numeriche)")
    plt.tight_layout()
    plt.show()


def compute_cramers_v(x, y):
    """
    Calcola il V di Cramer per due serie categoriche.
    :param x: Serie Pandas (variabile categorica).
    :param y: Serie Pandas (variabile categorica).
    :return: Valore del V di Cramer.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    # Correzione di bias
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def analyze_categorical_correlations(df, cat_features):
    """
    Calcola la matrice del V di Cramer per le variabili categoriche.
    :param df: DataFrame contenente i dati.
    :param cat_features: Lista di colonne categoriche.
    """
    cramers_matrix = pd.DataFrame(np.zeros((len(cat_features), len(cat_features))),
                                  index=cat_features, columns=cat_features)

    for col1 in cat_features:
        for col2 in cat_features:
            cramers_matrix.loc[col1, col2] = compute_cramers_v(df[col1], df[col2])

    print("=== Matrice del V di Cramer per le variabili categoriche ===")
    print(cramers_matrix, "\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap del V di Cramer (variabili categoriche)")
    plt.tight_layout()
    plt.show()


def perform_anova(df, num_features, cat_features):
    """
    Esegue ANOVA per valutare l'effetto delle variabili categoriche sulle variabili numeriche.
    Per ogni coppia (variabile numerica, variabile categorica) viene calcolato il test ANOVA.
    :param df: DataFrame contenente i dati.
    :param num_features: Lista di variabili numeriche.
    :param cat_features: Lista di variabili categoriche.
    """
    print("=== Risultati ANOVA (effetto delle variabili categoriche su quelle numeriche) ===")
    for num in num_features:
        for cat in cat_features:
            # Raggruppa i dati per la categoria corrente
            groups = [group[num].dropna().values for name, group in df.groupby(cat)]
            if len(groups) > 1:
                stat, p_value = f_oneway(*groups)
                print(f"ANOVA per {num} in base a {cat}: F = {stat:.2f}, p-value = {p_value:.4f}")
    print()


def one_hot_encoded_correlation(df, num_features, cat_features):
    """
    Converte le variabili categoriche in variabili dummy (One-Hot Encoding)
    e calcola la matrice di correlazione tra le variabili numeriche originali e quelle dummy.
    :param df: DataFrame contenente i dati.
    :param num_features: Lista di variabili numeriche.
    :param cat_features: Lista di variabili categoriche.
    """
    # One-Hot Encoding per le colonne categoriche
    df_dummies = pd.get_dummies(df[cat_features], drop_first=True)
    # Unione delle variabili numeriche e dummy
    df_combined = pd.concat([df[num_features], df_dummies], axis=1)
    corr = df_combined.corr()

    print("=== Matrice di correlazione (One-Hot Encoded per variabili categoriche) ===")
    print(corr, "\n")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
    plt.title("Heatmap delle correlazioni (numeriche e One-Hot Encoded)")
    plt.tight_layout()
    plt.show()


def chi_square_tests(df, cat_features, target):
    """
    Esegue il test del chi-quadro per verificare l'associazione tra le variabili categoriche e il target.
    :param df: DataFrame contenente i dati.
    :param cat_features: Lista di variabili categoriche.
    :param target: Nome della variabile target.
    """
    print("=== Risultati del test Chi-Quadrato per l'associazione con il target ===")
    for feature in cat_features:
        contingency_table = pd.crosstab(df[feature], df[target])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Test Chi-Quadrato tra {feature} e {target}: chi2 = {chi2:.2f}, p-value = {p:.4f}")
    print()


def preliminary_insights(df, num_features, cat_features):
    """
    Stampa alcuni insight preliminari e suggerimenti di pre-processing.
    - Outlier nelle variabili numeriche (verificabili con boxplot o analisi statistica).
    - Variabili numeriche con alta correlazione ridondante.
    - Variabili categoriche con classi poco rappresentate.
    Suggerimenti:
      - Imputazione dei valori mancanti (es. 'bmi')
      - Gestione delle categorie atipiche (es. 'gender' con "Other")
      - Bilanciamento della classe target 'stroke' (SMOTE o undersampling)
      - One-Hot Encoding per le variabili categoriche
    """
    print("=== Insight Preliminari e Strategie di Pre-processing ===")
    # Outlier: possiamo stampare le statistiche e suggerire di usare boxplot per l'analisi
    for feature in num_features:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature].count()
        print(f"{feature}: {n_outliers} possibili outlier rilevati (usando 1.5*IQR).")
    print()

    # Variabili con alta correlazione ridondante (es. correlazione > 0.8)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    redundant_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                redundant_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    if redundant_pairs:
        print("Variabili numeriche con correlazione > 0.8 (possibile ridondanza):")
        for pair in redundant_pairs:
            print(f" - {pair[0]} e {pair[1]}: correlazione = {pair[2]:.2f}")
    else:
        print("Non sono state rilevate correlazioni numeriche ridondanti significative.")
    print()

    # Variabili categoriche con classi poco rappresentate (es. count < 5)
    for feature in cat_features:
        counts = df[feature].value_counts()
        rare_classes = counts[counts < 5]
        if not rare_classes.empty:
            print(f"Variabile '{feature}' presenta classi con poche osservazioni:")
            print(rare_classes)
    print()

    print("Strategie suggerite:")
    print("- Imputare i valori mancanti, in particolare per 'bmi', utilizzando la mediana.")
    print("- Gestire le categorie atipiche, come la categoria 'Other' in 'gender'.")
    print("- Bilanciare la classe target 'stroke' mediante tecnica SMOTE.")
    print("- Il One-Hot Encoding è stato applicato per l'analisi della correlazione, ma dovrà essere rifatto nel pre-processing prima dell'addestramento del modello.\n")

# =============================================================================
# FUNZIONE PRINCIPALE
# =============================================================================

def eda(file_path):
    # 1) Caricamento del Dataset e Overview
    df = load_dataset(file_path)
    print("\n##############################")
    print("STEP 1: OVERVIEW DEL DATASET")
    print("##############################\n")
    overview_data(df)

    # 2) Analisi dei Valori Mancanti e Duplicati
    print("########################################")
    print("STEP 2: VALORI MANCANTI E RECORD DUPLICATI")
    print("########################################\n")
    analyze_missing_duplicates(df)

    # 3) Distribuzione delle Variabili
    print("########################################")
    print("STEP 3: DISTRIBUZIONE DELLE VARIABILI")
    print("########################################\n")
    # Variabili numeriche
    print("Colonne presenti nel DataFrame dopo il pre-processing:", df.columns)
    num_features = ['age', 'avg_glucose_level', 'bmi']
    plot_numeric_distributions(df, num_features)

    # Variabili categoriche (modifica in base alle colonne presenti nel dataset)
    cat_features = ['gender', 'work_type', 'smoking_status', 'ever_married', 'Residence_type',
                    'hypertension', 'heart_disease']
    plot_categorical_distributions(df, cat_features)

    # Distribuzione della classe target 'stroke'
    plot_target_distribution(df, 'stroke')

    # 4) Analisi delle Correlazioni
    print("########################################")
    print("STEP 4: ANALISI DELLE CORRELAZIONI")
    print("########################################\n")
    # 4a. Heatmap delle correlazioni numeriche
    plot_numeric_correlation_heatmap(df)

    # 4b. Cramér’s V per variabili categoriche
    analyze_categorical_correlations(df, cat_features)

    # 4c. ANOVA: effetto delle variabili categoriche sulle variabili numeriche
    perform_anova(df, num_features, cat_features)

    # 4d. One-Hot Encoding per correlazioni tra numeriche e categoriche
    one_hot_encoded_correlation(df, num_features, cat_features)

    # 4e. Test Chi-Quadrato per l'associazione tra variabili categoriche e il target 'stroke'
    chi_square_tests(df, cat_features, 'stroke')

    # 5) Insight Preliminari e Strategie di Pre-processing
    print("########################################")
    print("STEP 5: INSIGHT PRELIMINARI E STRATEGIE DI PRE-PROCESSING")
    print("########################################\n")
    preliminary_insights(df, num_features, cat_features)