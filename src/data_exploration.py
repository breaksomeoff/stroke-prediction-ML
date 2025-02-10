# data_exploration.py
# EDA per la predizione dell'ictus:
#
# Tutti i plot vengono salvati in "data/eda/plots" e il report in "data/eda/eda_report.txt"
# I plot per correlazioni (matrice numerica, Cramér's V) vengono salvati nella cartella "correlations"
# mentre i plot derivanti da ANOVA e Chi-Quadro sono salvati nella cartella "statistical_tests".

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency
from pyampute.exploration.mcar_statistical_tests import MCARTest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import logging
from scripts import config

# Impostazione logging di base
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

###############################################################################
#                               Utility Functions
###############################################################################

def cramers_v(x, y):
    """
    Calcolo di Cramér's V per stimare la correlazione tra variabili categoriche.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2corr / min(kcorr - 1, rcorr - 1))

def compute_cramers_v_matrix(df, cat_cols):
    n = len(cat_cols)
    results = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                results[i, j] = 1.0
            else:
                val = cramers_v(df[cat_cols[i]], df[cat_cols[j]])
                results[i, j] = val
                results[j, i] = val
    return pd.DataFrame(results, index=cat_cols, columns=cat_cols)

def compute_vif(df, numeric_cols):
    X = df[numeric_cols].dropna()
    if X.empty or X.shape[1] < 2:
        return pd.DataFrame({"feature": numeric_cols, "VIF": [np.nan] * len(numeric_cols)})
    X = sm.add_constant(X)
    vif_data = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif_value = variance_inflation_factor(X.values, i)
        vif_data.append((col, vif_value))
    return pd.DataFrame(vif_data, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

###############################################################################
#                       Modular Functions for Each Analysis
###############################################################################

def analyze_missing_values(data, report_file):
    """
    - Manteniamo il barplot dei missing e Little’s MCAR Test.
    """
    missing_plots_path = os.path.join(config.EDA_PLOTS_PATH, "missing_plots")
    os.makedirs(missing_plots_path, exist_ok=True)

    # Calcolo dei valori mancanti
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    high_missing_cols = missing_percent[missing_percent > 30].index.tolist()

    # Little’s MCAR Test
    numeric_only = data.select_dtypes(include=[np.number])
    mcar_test = MCARTest(method='little')
    p_value = mcar_test.little_mcar_test(numeric_only)

    # Creazione del barplot dei missing
    plt.figure(figsize=(6, 4))
    missing_percent.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Percentuale di Missing per Colonna")
    plt.ylabel("Percentuale (%)")
    plt.tight_layout()
    missing_barplot_path = os.path.join(missing_plots_path, "missing_barplot.png")
    plt.savefig(missing_barplot_path)
    plt.close()
    logging.info(f"[EDA] Missing barplot salvato in: {missing_barplot_path}")

    # Scrittura nel report
    report_file.write("=== Analisi Valori Mancanti ===\n")
    report_file.write("Valori mancanti (count):\n")
    report_file.write(str(missing_count) + "\n\n")
    report_file.write("Percentuale di valori mancanti:\n")
    report_file.write(str(missing_percent) + "\n\n")
    report_file.write(f"Colonne con >30% missing: {high_missing_cols}\n\n")
    report_file.write(f"Little’s MCAR Test p-value = {p_value:.6f}\n")
    interpretation = "MCAR" if p_value > 0.05 else "NON MCAR"
    report_file.write(f"Interpretazione: {interpretation}\n\n")
    logging.info("[EDA] Analisi valori mancanti completata.")

def plot_numeric_distributions(data, numeric_cols, report_file):
    """
    - Mantiene l'istogramma con hue per tutte le variabili numeriche.
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, impossibile plot numeriche.\n\n")
        return

    numeric_dist_path = os.path.join(config.EDA_PLOTS_PATH, "numeric_distributions")
    os.makedirs(numeric_dist_path, exist_ok=True)

    for col in numeric_cols:
        if col not in ["age", "avg_glucose_level", "bmi"]:
            # Creazione di un boxplot con hue solo per le altre variabili
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target_col, y=col, data=data, hue=target_col)
            plt.title(f"Boxplot di {col} vs {target_col}")
            plt.legend(loc='best')
            boxplot_path = os.path.join(numeric_dist_path, f"box_hue_{col}.png")
            plt.savefig(boxplot_path)
            plt.close()
            logging.info(f"[EDA] Boxplot salvato in: {boxplot_path}")

        # Istogramma con hue per ogni variabile numerica
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=col, hue=target_col, kde=True, element="step")
        plt.title(f"Histogram di {col} suddiviso per {target_col}")
        hist_path = os.path.join(numeric_dist_path, f"hist_{col}.png")
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"[EDA] Histogram salvato in: {hist_path}")

    report_file.write("=== Distribuzione Variabili Numeriche (Hist) ===\n")
    report_file.write(f"Plot salvati in: {numeric_dist_path}\n\n")
    logging.info("[EDA] Analisi distribuzione numeriche completata.")

def plot_categorical_distributions(data, cat_cols, report_file):
    """
    Per ogni variabile categorica (eccetto target), crea un histogram con hue=target (discrete=True).
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, nessun hist per categoriche.\n\n")
        return

    cat_dist_path = os.path.join(config.EDA_PLOTS_PATH, "categorical_distributions")
    os.makedirs(cat_dist_path, exist_ok=True)

    cols_for_hist = [c for c in cat_cols if c != target_col]
    for col in cols_for_hist:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=col, hue=target_col, multiple="stack", discrete=True)
        plt.title(f"Histogram categorico di {col} vs {target_col}")
        cat_hist_path = os.path.join(cat_dist_path, f"hist_categorical_{col}.png")
        plt.savefig(cat_hist_path)
        plt.close()
        logging.info(f"[EDA] Histogram categorico salvato in: {cat_hist_path}")

    report_file.write("=== Distribuzione Variabili Categoriche (Hist) ===\n")
    report_file.write(f"Plot salvati in: {cat_dist_path}\n\n")
    logging.info("[EDA] Analisi distribuzione categoriche completata.")

def analyze_target_imbalance(data, report_file):
    """
    Mantiene la Pie chart per il target.
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, impossibile analizzare squilibrio.\n\n")
        return

    target_imb_path = os.path.join(config.EDA_PLOTS_PATH, "target_imbalance")
    os.makedirs(target_imb_path, exist_ok=True)

    target_counts = data[target_col].value_counts(dropna=False)
    plt.figure(figsize=(6, 6))
    plt.pie(target_counts, labels=target_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
    plt.title("Distribuzione Target (Pie Chart)")
    pie_path = os.path.join(target_imb_path, "target_piechart.png")
    plt.savefig(pie_path)
    plt.close()
    logging.info(f"[EDA] Pie chart target salvata in: {pie_path}")

    report_file.write("=== Analisi dello Squilibrio del Target ===\n")
    report_file.write(f"Conteggio Target:\n{str(target_counts)}\n\n")

def analyze_correlations(data, numeric_cols, categorical_cols, report_file):
    """
    Analizza correlazioni numeriche (heatmap), Cramér's V per variabili categoriche e VIF.
    """
    corr_path = os.path.join(config.EDA_PLOTS_PATH, "correlations")
    os.makedirs(corr_path, exist_ok=True)

    if len(numeric_cols) > 1:
        corr_matrix_numeric = data[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix_numeric, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di Correlazione (Numeriche)")
        corr_numeric_path = os.path.join(corr_path, "corr_matrix_numeric.png")
        plt.savefig(corr_numeric_path)
        plt.close()
        logging.info(f"[EDA] Matrice di correlazione numerica salvata in: {corr_numeric_path}")
    else:
        corr_matrix_numeric = "Non abbastanza col. numeriche."

    if len(categorical_cols) > 1:
        cramer_matrix = compute_cramers_v_matrix(data, categorical_cols)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cramer_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di Cramér's V (Categoriche)")
        cramers_path = os.path.join(corr_path, "cramers_v_matrix.png")
        plt.savefig(cramers_path)
        plt.close()
        logging.info(f"[EDA] Matrice di Cramér's V salvata in: {cramers_path}")
    else:
        cramer_matrix = "Non abbastanza col. categoriche."

    vif_df = compute_vif(data, numeric_cols)

    report_file.write("=== Correlazioni (Numeriche, Cramér's V, VIF) ===\n")
    report_file.write(">> Matrice Numerica:\n")
    report_file.write(str(corr_matrix_numeric) + "\n\n")
    report_file.write(">> Cramér's V (Categoriche):\n")
    report_file.write(str(cramer_matrix) + "\n\n")
    report_file.write(">> VIF (Multicollinearità):\n")
    report_file.write(str(vif_df) + "\n\n")
    report_file.write(f"Grafici salvati in: {corr_path}\n\n")
    logging.info("[EDA] Analisi correlazioni completata.")

def analyze_outliers(data, numeric_cols, report_file):
    """
    Analizza outlier e genera scatter plot per ogni variabile numerica.
    """
    outlier_path = os.path.join(config.EDA_PLOTS_PATH, "outliers")
    os.makedirs(outlier_path, exist_ok=True)

    outlier_summary = {}
    for col in numeric_cols:
        col_data = data[col].dropna()
        if len(col_data) == 0:
            continue
        Q1, Q3 = col_data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lb = Q1 - 1.5 * IQR
        ub = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lb) | (col_data > ub)]

        outlier_summary[col] = {
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lb,
            "upper_bound": ub,
            "num_outliers": len(outliers),
            "percent_outliers": (len(outliers) / len(col_data)) * 100
        }

        if config.TARGET_COLUMN in data.columns and col != config.TARGET_COLUMN:
            plt.figure(figsize=(6, 4))
            if data[config.TARGET_COLUMN].dtype in [int, float]:
                sns.scatterplot(x=data[col], y=data[config.TARGET_COLUMN])
                plt.title(f"Scatter: {col} vs {config.TARGET_COLUMN}")
            else:
                sns.scatterplot(
                    x=data[col],
                    y=np.random.normal(size=len(data)),
                    hue=data[config.TARGET_COLUMN],
                    palette="Set1"
                )
                plt.title(f"Scatter: {col} (x) hue={config.TARGET_COLUMN}")
            scatter_path = os.path.join(outlier_path, f"scatter_outliers_{col}.png")
            plt.savefig(scatter_path)
            plt.close()
            logging.info(f"[EDA] Scatter plot outlier per {col} salvato in: {scatter_path}")

    report_file.write("=== Outlier Analysis ===\n")
    for col, info in outlier_summary.items():
        report_file.write(
            f"  * {col}: IQR={info['IQR']:.2f}, "
            f"Bounds=({info['lower_bound']:.2f}, {info['upper_bound']:.2f}), "
            f"Outliers={info['num_outliers']}, "
            f"PercOut={info['percent_outliers']:.2f}%\n"
        )
    logging.info("[EDA] Analisi outlier completata.")

###############################################################################
#                Funzioni per ANOVA e Chi-Quadrato con Plot
###############################################################################

def plot_anova_results(anova_res, report_file):
    """
    anova_res: lista di tuple (col, f_stat, p_val) per feature numeriche.
    Genera un barplot in base a -log10(p_val).
    """
    if not anova_res:
        report_file.write("Nessun risultato ANOVA disponibile (target assente o monotono).\n")
        return

    # Creiamo una cartella dedicata per i plot statistici
    stat_tests_path = os.path.join(config.EDA_PLOTS_PATH, "statistical_tests")
    os.makedirs(stat_tests_path, exist_ok=True)

    anova_df = pd.DataFrame(anova_res, columns=["feature", "f_stat", "p_val"])
    anova_df["importance"] = -np.log10(anova_df["p_val"] + 1e-16)
    anova_df.sort_values("importance", ascending=False, inplace=True)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=anova_df, x="importance", y="feature", color="skyblue", edgecolor="black")
    plt.xlabel("-log10(p-value) [ANOVA]")
    plt.ylabel("Feature")
    plt.title("ANOVA Importance (Numeriche)")
    anova_plot_path = os.path.join(stat_tests_path, "anova_plot.png")
    plt.savefig(anova_plot_path)
    plt.close()
    report_file.write(f"[ANOVA] Plot generato: {anova_plot_path}\n")
    logging.info(f"[EDA] Plot ANOVA salvato in: {anova_plot_path}")

def plot_chi2_results(chi2_res, report_file):
    """
    chi2_res: lista di tuple (col, chi2_stat, p_val) per feature categoriche.
    Genera un barplot in base a -log10(p_val).
    """
    if not chi2_res:
        report_file.write("Nessun risultato Chi-Quadro disponibile (target assente o monotono).\n")
        return

    # Utilizziamo la stessa cartella dei test statistici
    stat_tests_path = os.path.join(config.EDA_PLOTS_PATH, "statistical_tests")
    os.makedirs(stat_tests_path, exist_ok=True)

    chi2_df = pd.DataFrame(chi2_res, columns=["feature", "chi2_stat", "p_val"])
    chi2_df["importance"] = -np.log10(chi2_df["p_val"] + 1e-16)
    chi2_df.sort_values("importance", ascending=False, inplace=True)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=chi2_df, x="importance", y="feature", color="lightgreen", edgecolor="black")
    plt.xlabel("-log10(p-value) [Chi2]")
    plt.ylabel("Feature")
    plt.title("Chi-Quadrato Importance (Categoriche)")
    chi2_plot_path = os.path.join(stat_tests_path, "chi2_plot.png")
    plt.savefig(chi2_plot_path)
    plt.close()
    report_file.write(f"[Chi2] Plot generato: {chi2_plot_path}\n")
    logging.info(f"[EDA] Plot Chi2 salvato in: {chi2_plot_path}")

###############################################################################
#                               Main Function
###############################################################################

def main():
    data = pd.read_csv(config.RAW_DATA_PATH)
    os.makedirs(config.EDA_PLOTS_PATH, exist_ok=True)

    with open(config.EDA_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("===== EDA REPORT MODIFICATO =====\n\n")

        # 1) Analisi Valori Mancanti
        analyze_missing_values(data, report_file)

        # Distinzione numeric/categorical
        numeric_cols = ["age", "bmi", "avg_glucose_level"]
        categorical_cols = ["gender", "hypertension", "heart_disease", "ever_married",
                            "work_type", "Residence_type", "smoking_status", "stroke"]

        # 2) Distribuzione Numeriche
        plot_numeric_distributions(data, numeric_cols, report_file)

        # 2.b) Distribuzione Categoriche (solo hist)
        plot_categorical_distributions(data, categorical_cols, report_file)

        # 3) Analisi dello Squilibrio del Target (Pie chart)
        analyze_target_imbalance(data, report_file)

        # 4) Correlazioni
        analyze_correlations(data, numeric_cols, categorical_cols, report_file)

        # 5) Outlier Analysis
        analyze_outliers(data, numeric_cols, report_file)

        # 6) ANOVA / Chi-Quadro con i relativi plot
        report_file.write("=== ANOVA / Chi-Quadro ===\n")
        target_col = config.TARGET_COLUMN
        anova_res = []
        chi2_res = []
        if target_col in data.columns and data[target_col].nunique() > 1:
            # ANOVA per variabili numeriche
            for col in numeric_cols:
                if col == target_col:
                    continue
                groups = [data[data[target_col] == c][col].dropna() for c in data[target_col].unique()]
                if len(groups) < 2 or any(len(g) == 0 for g in groups):
                    continue
                f_stat, p_val = f_oneway(*groups)
                anova_res.append((col, f_stat, p_val))

            # Chi-Quadrato per variabili categoriche
            for col in categorical_cols:
                if col == target_col:
                    continue
                ct = pd.crosstab(data[col], data[target_col])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2_stat, p_val_chi, _, _ = chi2_contingency(ct)
                chi2_res.append((col, chi2_stat, p_val_chi))

            report_file.write("ANOVA (numeriche):\n")
            for col, f_stat, p_val in anova_res:
                report_file.write(f" - {col}: F={f_stat:.4f}, p={p_val:.6f}\n")
            report_file.write("Chi-Quadro (categoriche):\n")
            for col, chi2_stat, p_val_chi in chi2_res:
                report_file.write(f" - {col}: Chi2={chi2_stat:.4f}, p={p_val_chi:.6f}\n")
            report_file.write("\n")

            # Generazione dei plot per ANOVA e Chi2
            plot_anova_results(anova_res, report_file)
            plot_chi2_results(chi2_res, report_file)
        else:
            report_file.write("Target non presente o monotono. ANOVA / Chi2 non calcolati.\n\n")

        report_file.write("===== FINE EDA REPORT =====\n")

    logging.info(f"[SUCCESS] EDA completata con successo. Plot salvati in: {config.EDA_PLOTS_PATH}, Report in: {config.EDA_REPORT_PATH}")

if __name__ == "__main__":
    main()
