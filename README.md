## 🏥 Stroke Prediction ML 🏥  
_Un sistema di machine learning per la predizione del rischio di ictus_  

### 📌 Descrizione  
Questo progetto utilizza tecniche di **Machine Learning** per prevedere il rischio di **ictus** nei pazienti, basandosi su dati clinici e demografici. L'obiettivo è sviluppare un modello di classificazione in grado di identificare soggetti a rischio, fornendo un aiuto nella prevenzione e nella diagnosi precoce.  

Il progetto include l'intero flusso di sviluppo di un modello ML, dalla **preparazione dei dati** all'**addestramento del modello**, fino alla **valutazione delle performance**.  

---

### 📂 Struttura della repository  

```
stroke-prediction-ML/
│── 📜 README.md                     # Documentazione del progetto
│── 📜 requirements.txt              # Librerie necessarie
│── 📜 .gitignore                    # File ignorati da Git
│── 📜 LICENSE                       # Licenza del progetto
│── 📂 data/                         # Dati grezzi e pre-processati
│   │── 📜 raw/                      # Dataset originale
│   │── 📜 processed/                # Dataset dopo il pre-processing
│── 📂 notebooks/                    # Jupyter Notebooks per analisi e sviluppo
│   │── 📜 01_EDA.ipynb              # Analisi esplorativa dei dati
│   │── 📜 02_Preprocessing.ipynb    # Pre-processing & Feature Engineering
│   │── 📜 03_Training.ipynb         # Addestramento del modello
│   │── 📜 04_Evaluation.ipynb       # Valutazione e metriche
│── 📂 src/                          # Codice principale del progetto
│   │── 📜 data_preprocessing.py     # Pulizia e preparazione dati
│   │── 📜 feature_engineering.py    # Creazione e selezione feature
│   │── 📜 train_model.py            # Addestramento del modello
│   │── 📜 evaluate_model.py         # Valutazione delle prestazioni
│── 📂 models/                       # Modelli addestrati e metriche                    
└── 📂 scripts/                      # Script per eseguire pipeline
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
✔ **Scikit-Learn, XGBoost** (Modelli di Machine Learning)  
✔ **Matplotlib, Seaborn** (Visualizzazione dati)  
✔ **SHAP, LIME** (Interpretabilità del modello)  
✔ **Jupyter Notebook**  

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

#### ⚡ **3. Eseguire il progetto**
- **Analisi esplorativa**:
  ```bash
  jupyter notebook notebooks/01_EDA.ipynb
  ```
- **Pre-processing e Feature Engineering**:
  ```bash
  python src/data_preprocessing.py
  ```
- **Addestrare il modello**:
  ```bash
  python src/train_model.py --model xgboost
  ```
- **Valutazione del modello**:
  ```bash
  python src/evaluate_model.py
  ```

---

### 📈 Valutazione del modello  
I modelli saranno valutati utilizzando le seguenti metriche:  
✔ **Accuracy**  
✔ **Precision, Recall, F1-score** (per gestire il bilanciamento delle classi)  
✔ **ROC-AUC Score**  
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