# data_exploration.py
# EDA per la predizione dell'ictus (versione con modifiche richieste):
#   - Rimosse: dynamic pairplot, missing heatmap, missing vs target,
#              boxplot con hue per age/avg_glucose_level/bmi, donut chart del target.
#   - Aggiunti:
#       * Hist per feature categoriche
#       * Barplot ANOVA (numeriche)
#       * Barplot Chi-Quadro (categoriche)

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency
from pyampute.exploration.mcar_statistical_tests import MCARTest
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scripts import config

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
    phi2corr = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))
    rcorr = r - (r - 1)**2 / (n - 1)
    kcorr = k - (k - 1)**2 / (n - 1)
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
        return pd.DataFrame({"feature": numeric_cols, "VIF": [np.nan]*len(numeric_cols)})
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
    - Rimosso: heatmap e missing_vs_target come da richiesta.
    - Manteniamo solo barplot e Little’s MCAR Test.
    """
    missing_plots_path = os.path.join(config.EDA_PLOTS_PATH, "missing_plots")
    os.makedirs(missing_plots_path, exist_ok=True)

    # Conteggio e percentuale
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    high_missing_cols = missing_percent[missing_percent > 30].index.tolist()

    # MCAR Test
    numeric_only = data.select_dtypes(include=[np.number])
    mcar_test = MCARTest(method='little')
    p_value = mcar_test.little_mcar_test(numeric_only)

    # Bar plot (unico rimasto)
    plt.figure(figsize=(6, 4))
    missing_percent.sort_values(ascending=False).plot(
        kind='bar', color='skyblue', edgecolor='black')
    plt.title("Percentuale di Missing per Colonna")
    plt.ylabel("Percentuale (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(missing_plots_path, "missing_barplot.png"))
    plt.close()

    # Scrittura nel report
    report_file.write("=== Analisi Valori Mancanti ===\n")
    report_file.write("Valori mancanti (count):\n")
    report_file.write(str(missing_count) + "\n\n")
    report_file.write("Percentuale di valori mancanti:\n")
    report_file.write(str(missing_percent) + "\n\n")
    report_file.write(f"Colonne con più del 30% di valori mancanti: {high_missing_cols}\n\n")
    report_file.write(f"Little’s MCAR Test p-value = {p_value:.6f}\n")
    interpretation = "MCAR" if p_value > 0.05 else "NON MCAR"
    report_file.write(f"Interpretazione: {interpretation}\n\n")

def plot_numeric_distributions(data, numeric_cols, report_file):
    """
    - Rimosso box_hue_{age,avg_glucose_level,bmi} (quindi niente boxplot per quelle 3)
      ma se la richiesta dice "rimuovi i box hue solo per age, avg_glucose_level, bmi",
      possiamo farlo con un semplice if:
    - Manteniamo l'istogramma con hue comunque.
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, impossibile plot numeriche.\n\n")
        return

    numeric_dist_path = os.path.join(config.EDA_PLOTS_PATH, "numeric_distributions")
    os.makedirs(numeric_dist_path, exist_ok=True)

    for col in numeric_cols:
        # Se la colonna è tra age, avg_glucose_level, bmi => NIENTE boxplot
        if col not in ["age", "avg_glucose_level", "bmi"]:
            # boxplot con hue se la colonna è numerica ma NON è age/avg_glucose_level/bmi
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=target_col, y=col, data=data, hue=target_col)
            plt.title(f"Boxplot di {col} vs {target_col}")
            plt.legend(loc='best')
            plt.savefig(os.path.join(numeric_dist_path, f"box_hue_{col}.png"))
            plt.close()

        # Istogramma con hue (sempre)
        plt.figure(figsize=(6, 4))
        sns.histplot(data=data, x=col, hue=target_col, kde=True, element="step")
        plt.title(f"Histogram di {col} suddiviso per {target_col}")
        plt.savefig(os.path.join(numeric_dist_path, f"hist_{col}.png"))
        plt.close()

    report_file.write("=== Distribuzione Variabili Numeriche ===\n")
    report_file.write("- Boxplot con hue rimosso per age, avg_glucose_level, bmi\n")
    report_file.write("- Rimasti solo histogram con hue per tutte\n")
    report_file.write(f"Plot salvati in: {numeric_dist_path}\n\n")

def plot_categorical_distributions(data, cat_cols, report_file):
    """
    Nuova funzione:
    - Per ogni variabile categorica (eccetto target), crea un hist con hue=target
      (se target esiste).
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, nessun hist per categoriche.\n\n")
        return

    cat_dist_path = os.path.join(config.EDA_PLOTS_PATH, "categorical_distributions")
    os.makedirs(cat_dist_path, exist_ok=True)

    # Evitiamo di fare hist su 'stroke' stesso
    cols_for_hist = [c for c in cat_cols if c != target_col]

    for col in cols_for_hist:
        plt.figure(figsize=(6, 4))
        # Convertiamo la colonna categorica in string codes?
        # Oppure usiamo countplot. Ma l'utente ha detto "solo hist".
        # Per hist su categoriche, possiamo forzare discrete=True:
        sns.histplot(data=data, x=col, hue=target_col, multiple="stack", discrete=True)
        plt.title(f"Histogram categorico di {col} vs {target_col}")
        plt.savefig(os.path.join(cat_dist_path, f"hist_categorical_{col}.png"))
        plt.close()

    report_file.write("=== Distribuzione Variabili Categoriche (Hist) ===\n")
    report_file.write(f"Plot salvati in: {cat_dist_path}\n\n")

def analyze_target_imbalance(data, report_file):
    """
    Rimosso il donut chart.
    Manteniamo solo la Pie chart.
    """
    target_col = config.TARGET_COLUMN
    if target_col not in data.columns:
        report_file.write("Target non presente, impossibile analizzare squilibrio.\n\n")
        return

    target_imb_path = os.path.join(config.EDA_PLOTS_PATH, "target_imbalance")
    os.makedirs(target_imb_path, exist_ok=True)

    target_counts = data[target_col].value_counts(dropna=False)

    # Pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(target_counts, labels=target_counts.index.astype(str), autopct='%1.1f%%', startangle=140)
    plt.title("Distribuzione Target (Pie Chart)")
    plt.savefig(os.path.join(target_imb_path, "target_piechart.png"))
    plt.close()

    report_file.write("=== Analisi dello Squilibrio del Target ===\n")
    report_file.write(f"Conteggio Target:\n{str(target_counts)}\n\n")

def analyze_correlations(data, numeric_cols, categorical_cols, report_file):
    corr_path = os.path.join(config.EDA_PLOTS_PATH, "correlations")
    os.makedirs(corr_path, exist_ok=True)

    if len(numeric_cols) > 1:
        corr_matrix_numeric = data[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix_numeric, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di Correlazione (Numeriche)")
        plt.savefig(os.path.join(corr_path, "corr_matrix_numeric.png"))
        plt.close()
    else:
        corr_matrix_numeric = "Non abbastanza col. numeriche."

    if len(categorical_cols) > 1:
        cramer_matrix = compute_cramers_v_matrix(data, categorical_cols)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cramer_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di Cramér's V (Categoriche)")
        plt.savefig(os.path.join(corr_path, "cramers_v_matrix.png"))
        plt.close()
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

def analyze_outliers(data, numeric_cols, report_file):
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
            plt.savefig(os.path.join(outlier_path, f"scatter_outliers_{col}.png"))
            plt.close()

    report_file.write("=== Outlier Analysis ===\n")
    for col, info in outlier_summary.items():
        report_file.write(
            f"  * {col}: IQR={info['IQR']:.2f}, "
            f"Bounds=({info['lower_bound']:.2f}, {info['upper_bound']:.2f}), "
            f"Outliers={info['num_outliers']}, "
            f"PercOut={info['percent_outliers']:.2f}%\n"
        )
    report_file.write("\nRaccomandazioni:\n")
    report_file.write("- Verificare la natura degli outlier (errori vs valori estremi reali).\n")
    report_file.write("- Valutare rimozione, trasformazioni (es. log) o winsorizing.\n\n")

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

    anova_df = pd.DataFrame(anova_res, columns=["feature", "f_stat", "p_val"])
    # Creiamo una colonna di importanza
    anova_df["importance"] = -np.log10(anova_df["p_val"] + 1e-16)  # Evitiamo log(0)
    anova_df.sort_values("importance", ascending=False, inplace=True)

    # Barplot
    plt.figure(figsize=(8, 4))
    sns.barplot(data=anova_df, x="importance", y="feature", color="skyblue", edgecolor="black")
    plt.xlabel("-log10(p-value) [ANOVA]")
    plt.ylabel("Feature")
    plt.title("ANOVA Importance (Numeriche)")
    anova_plot_path = os.path.join(config.EDA_PLOTS_PATH, "anova_plot.png")
    plt.savefig(anova_plot_path)
    plt.close()

    report_file.write(f"[ANOVA] Plot generato: {anova_plot_path}\n")

def plot_chi2_results(chi2_res, report_file):
    """
    chi2_res: lista di tuple (col, chi2_stat, p_val) per feature categoriche.
    Genera un barplot in base a -log10(p_val).
    """
    if not chi2_res:
        report_file.write("Nessun risultato Chi-Quadro disponibile (target assente o monotono).\n")
        return

    chi2_df = pd.DataFrame(chi2_res, columns=["feature", "chi2_stat", "p_val"])
    chi2_df["importance"] = -np.log10(chi2_df["p_val"] + 1e-16)
    chi2_df.sort_values("importance", ascending=False, inplace=True)

    # Barplot
    plt.figure(figsize=(8, 4))
    sns.barplot(data=chi2_df, x="importance", y="feature", color="lightgreen", edgecolor="black")
    plt.xlabel("-log10(p-value) [Chi2]")
    plt.ylabel("Feature")
    plt.title("Chi-Quadrato Importance (Categoriche)")
    chi2_plot_path = os.path.join(config.EDA_PLOTS_PATH, "chi2_plot.png")
    plt.savefig(chi2_plot_path)
    plt.close()

    report_file.write(f"[Chi2] Plot generato: {chi2_plot_path}\n")

###############################################################################
#                               Main Function
###############################################################################

def main():
    # Caricamento dataset
    data = pd.read_csv(config.RAW_DATA_PATH)

    os.makedirs(config.EDA_PLOTS_PATH, exist_ok=True)

    with open(config.EDA_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("===== EDA REPORT MODIFICATO =====\n\n")

        # 1) Missing
        analyze_missing_values(data, report_file)

        # Distinzione numeric/categorical
        numeric_cols = ["age", "bmi", "avg_glucose_level"]
        categorical_cols = ["gender", "hypertension", "heart_disease", "ever_married",
                            "work_type", "Residence_type", "smoking_status", "stroke"]

        # 2) Distribuzione Numeriche
        plot_numeric_distributions(data, numeric_cols, report_file)

        # 2.b) Distribuzione Categoriche (solo hist)
        plot_categorical_distributions(data, categorical_cols, report_file)

        # 3) Target imbalance (Pie chart)
        analyze_target_imbalance(data, report_file)

        # 4) Correlazioni
        analyze_correlations(data, numeric_cols, categorical_cols, report_file)

        # 5) Outlier
        analyze_outliers(data, numeric_cols, report_file)

        # 6) Rimosso dynamic_pairplot come da richiesta

        # 7) ANOVA / Chi2 con i relativi plot
        report_file.write("=== ANOVA / Chi-Quadro ===\n")
        target_col = config.TARGET_COLUMN
        anova_res = []
        chi2_res = []
        if target_col in data.columns and data[target_col].nunique() > 1:
            # ANOVA numeriche
            for col in numeric_cols:
                if col == target_col:
                    continue
                groups = [data[data[target_col] == c][col].dropna()
                          for c in data[target_col].unique()]
                # Evitiamo col. monotone
                if len(groups) < 2 or any(len(g) == 0 for g in groups):
                    continue
                f_stat, p_val = f_oneway(*groups)
                anova_res.append((col, f_stat, p_val))

            # Chi2 categoriche
            for col in categorical_cols:
                if col == target_col:
                    continue
                ct = pd.crosstab(data[col], data[target_col])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2_stat, p_val_chi, _, _ = chi2_contingency(ct)
                chi2_res.append((col, chi2_stat, p_val_chi))

            # Scrittura testuale
            report_file.write("ANOVA (numeriche):\n")
            for col, f_stat, p_val in anova_res:
                report_file.write(f" - {col}: F={f_stat:.4f}, p={p_val:.6f}\n")

            report_file.write("Chi-Quadro (categoriche):\n")
            for col, chi2_stat, p_val_chi in chi2_res:
                report_file.write(f" - {col}: Chi2={chi2_stat:.4f}, p={p_val_chi:.6f}\n")
            report_file.write("\n")

            # Generiamo i plot
            plot_anova_results([(c, f, p) for (c, f, p) in anova_res], report_file)
            plot_chi2_results([(c, chi, p) for (c, chi, p) in chi2_res], report_file)

        else:
            report_file.write("Target non presente o monotono. ANOVA / Chi2 non calcolati.\n\n")

        # 8) Raccomandazioni finali
        report_file.write("=== Raccomandazioni Finali ===\n")
        report_file.write("- Se MCAR: imputazione semplice; altrimenti MAR/MNAR.\n")
        report_file.write("- Colonne con >30% missing: considerare rimozione/imputazione.\n")
        report_file.write("- Outlier: trasformazioni (log) o winsorizing.\n")
        report_file.write("- VIF alto -> rimuovere/combinare feature collineari.\n")
        report_file.write("- Feature con p-value basso -> rilevanti per il modello.\n")
        report_file.write("- Target squilibrato? considerare SMOTE/undersampling.\n\n")

        report_file.write("===== FINE EDA REPORT =====\n")

    print(f"[INFO] EDA completata con successo. Plot in: {config.EDA_PLOTS_PATH}, Report in: {config.EDA_REPORT_PATH}")


if __name__ == "__main__":
    main()
