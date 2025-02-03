from src.data_exploration import eda
from src.data_preprocessing import preprocess_data
from src.train_test_smote import train_test_smote
from src.train_model import train_model
#from src.evaluate_model import evaluate_model

# Definire i percorsi
RAW_DATA_PATH = "../data/raw/stroke-data.csv"
PROCESSED_DATA_PATH = "../data/processed/processed-stroke-data.csv"
TRAIN_DATA_PATH = "../data/processed/train-stroke-data.csv"
TEST_DATA_PATH = "../data/processed/test-stroke-data.csv"
MODEL_PATH = "../models/stroke-model.pkl"

def main():
    """
    Esegue l'intera pipeline di Machine Learning:
    1) Esplorazione dei dati
    2) Pre-processing del dataset
    3) Split dei dati in Training e Test
    4) Addestramento del modello
    5) Validazione del modello
    6) Valutazione delle performance
    """
    print("\n🔎 STEP 1: Esplorazione dei dati...")
    eda(RAW_DATA_PATH)

    print("\n🚀 STEP 2: Pre-processing del dataset...")
    preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)  # Salva automaticamente il file pre-processato

    print("\n🔄 STEP 3: Split dei dati in Train (+ SMOTE) e Test...")
    train_test_smote(PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)

    """
    print("\n🎯 STEP 4: Addestramento del modello...")
    train_model(TRAIN_DATA_PATH, MODEL_PATH)
    
    print("\n🔍 STEP 5: Validazione del modello...")
    #attenzione al path assegnato
    validate_model(MODEL_PATH)

    print("\n📊 STEP 6: Valutazione del modello...")
    evaluate_model(TEST_DATA_PATH, MODEL_PATH)
    """

    print("\n✅ Pipeline completata con successo!")

#Esecuzione dell'intera pipeline di ML con:
if __name__ == "__main__":
    #1) Esplorazione iniziale del DS
    #eda(RAW_DATA_PATH)
    #2) Pre-processing DS ed encoding
    #preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    #3) Esplorazione post pre-processing
    #eda(PROCESSED_DATA_PATH)
    #4) Suddivisione in train e test e applicazione tecnica SMOTE per bilanciare le istanze della classe
    # target meno rappresentata
    train_test_smote(PROCESSED_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH)
    #5) Training del modello scelto attraverso la stringa passata alla funzione
    train_model()