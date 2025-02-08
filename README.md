## 🏥 Stroke Prediction ML 🏥  
_Un sistema di machine learning per la predizione del rischio di ictus_  

### 📌 Descrizione  
Questo progetto utilizza tecniche di **Machine Learning** per prevedere il rischio di **ictus** nei pazienti, basandosi su dati clinici e demografici. L'obiettivo è sviluppare un modello di classificazione in grado di identificare soggetti a rischio, fornendo un aiuto nella prevenzione e nella diagnosi precoce.  

Il progetto include l'intero flusso di sviluppo di un modello ML, dalla **preparazione dei dati** all'**addestramento del modello**, fino alla **valutazione delle performance**.  

---

### 📂 Struttura della repository  

```
stroke-prediction-ML/
│── 📜 README.md                  # README del progetto
│── 📜 requirements.txt           # Librerie necessarie per l'ambiente virtuale (venv)
│── 📜 .gitignore                 # File e cartelle ignorati da Git
│── 📜 LICENSE.md                 # Licenza del progetto
│── 📂 data/                      # Cartella per i dati grezzi e pre-processati
│   │── 📂 eda                    # File della Data Exploration
│   │── 📂 raw                    # Dataset originale
│   │── 📂 processed              # Dataset dopo il pre-processing
│── 📂 model/                     # Modello addestrato e salvato in formato .joblib
│   │── 📂 plots                  # Vari plots del modello
│   │── 📜 evaluation_report.txt  # Report delle metriche del modello
│   │── 📜 optimal_threshold.txt  # Valore ottimale della threshold del modello
│   │── 📜 rf-model.joblib        # Modello addestrato con Random Forest
│── 📂 scripts/                   # Script principali per eseguire pipeline e configurazione
│   │── 📜 config.py              # Configurazione generale del progetto (percorsi, parametri, ecc.)
│   │── 📜 run_pipeline.py        # Script principale per eseguire l'intera pipeline di ML
│── 📂 src/                       # Codice principale del progetto
│   │── 📜 data_exploration.py    # Analisi esplorativa dei dati (EDA)
│   │── 📜 data_preprocessing.py  # Pulizia, trasformazione e pre-processing dei dati
│   │── 📜 model.py               # Definizione, training e salvataggio del modello di Machine Learning
│   │── 📜 evaluation.py          # Valutazione del modello attraverso metriche e grafici
```

---

### 📊 Dataset  
Il dataset utilizzato è disponibile su **Kaggle**:  
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  

**Variabili principali:**  
- `gender`: Sesso del paziente  
- `age`: Età  
- `hypertension`: Presenza di ipertensione  
- `heart_disease`: Presenza di malattie cardiache  
- `ever_married`: Stato matrimoniale  
- `work_type`: Tipo di lavoro  
- `Residence_type`: Tipo di residenza  
- `avg_glucose_level`: Livello medio di glucosio  
- `bmi`: Indice di massa corporea  
- `smoking_status`: Stato di fumatore  
- `stroke`: **Target** (1 = ictus, 0 = nessun ictus)  

---

### 🏗️ Tecnologie utilizzate  
✔ **Python 3.x**  
✔ **Pandas, NumPy** (Analisi dati)  
✔ **Scikit-Learn** (Modelli di Machine Learning)  
✔ **Matplotlib, Seaborn** (Visualizzazione dati)  
---

### 🚀 Installazione e utilizzo  

#### 📥 **1. Clonare la repository**
```bash
git clone https://github.com/tuo-username/stroke-prediction-ML.git
cd stroke-prediction-ML
```

#### 📦 **2. Installare i pacchetti richiesti**
```bash
pip install -r requirements.txt
```

### 📈 3. Esecuzione e Valutazione del modello  
Avviare la pipeline tramite il comando  
```bash
py run_pipeline.py
```
Il modello sarà valutato utilizzando le seguenti metriche:  
✔ **Accuracy**  
✔ **Precision, Recall, F1-score** (per gestire il bilanciamento delle classi)  
✔ **ROC-AUC Curve**  
✔ **Precision-Recall Curve**  
✔ **Confusion Matrix**  

---

### 📜 Licenza  
Questo progetto è rilasciato sotto licenza **MIT**.  

---

### ⭐ Contatti  
- **Autori**:   
- [Inglese Alessio](https://github.com/breakesomeoff)
- [De Vita Adriano](https://github.com/adry04)
- 📩 Email: 
- [a.inglese11@studenti.unisa.it] 
- [a.devita40@studenti.unisa.it] 