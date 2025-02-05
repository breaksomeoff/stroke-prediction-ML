from src.data_preprocessing import preprocess_datasets
from src.train_test_smote import process_and_split
from src.hyperparameter_tuning import tune_models
from src.validate_model import validate_models
from src.evaluate_models import evaluate_models

# Definire i percorsi
RAW_DATA_PATH = "../data/raw/stroke-data.csv"

RANDOM_FOREST_PROCESSED_DATA_PATH = "../data/processed/processed-rf-stroke-data.csv"
GRADIENT_BOOSTING_PROCESSED_DATA_PATH = "../data/processed/processed-gb-stroke-data.csv"

RANDOM_FOREST_TRAIN_DATA_PATH = "../data/processed/train-rf-stroke-data.csv"
GRADIENT_BOOSTING_TRAIN_DATA_PATH = "../data/processed/train-gb-stroke-data.csv"

RANDOM_FOREST_TEST_DATA_PATH = "../data/processed/test-rf-stroke-data.csv"
GRADIENT_BOOSTING_TEST_DATA_PATH = "../data/processed/test-gb-stroke-data.csv"

RANDOM_FOREST_MODEL_PATH = "../models/rf-model.pkl"
GRADIENT_BOOSTING_MODEL_PATH = "../models/gb-model.pkl"

def main():
    """
    Esegue l'intera pipeline di Machine Learning:
    1) Esplorazione dei dati
    2) Pre-processing del dataset
    3) Split dei dati in Training e Test
    4) Addestramento del modello sul Training set
    5) Validazione del modello con K-Folds Cross Validation
    6) Valutazione delle performance sul Test set

    print("\n🔎 STEP 1: Esplorazione dei dati...")
    eda(RAW_DATA_PATH)
    """
    print("\n🚀 STEP 2: Pre-processing dei datasets...")
    preprocess_datasets(RAW_DATA_PATH, RANDOM_FOREST_PROCESSED_DATA_PATH, GRADIENT_BOOSTING_PROCESSED_DATA_PATH)

    print("\n🔄 STEP 3: Split dei datasets pre-processati in Train (+ SMOTE) e Test...")
    process_and_split(RANDOM_FOREST_PROCESSED_DATA_PATH, GRADIENT_BOOSTING_PROCESSED_DATA_PATH, RANDOM_FOREST_TRAIN_DATA_PATH, RANDOM_FOREST_TEST_DATA_PATH, GRADIENT_BOOSTING_TRAIN_DATA_PATH, GRADIENT_BOOSTING_TEST_DATA_PATH)

    print("\n🎯 STEP 4: Hyperparameter Tuning dei modelli...")
    tune_models(RANDOM_FOREST_TRAIN_DATA_PATH, GRADIENT_BOOSTING_TRAIN_DATA_PATH, RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH, n_iter=50)

    print("\n🔍 STEP 5: Validazione dei modelli...")
    validate_models(RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH, RANDOM_FOREST_TRAIN_DATA_PATH, GRADIENT_BOOSTING_TRAIN_DATA_PATH, k=10)

    print("\n📊 STEP 6: Valutazione e confronto di entrambi i modelli...")
    evaluate_models(RANDOM_FOREST_TEST_DATA_PATH, GRADIENT_BOOSTING_TEST_DATA_PATH, RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH)

    print("\n✅ Pipeline completata con successo!")

#Esecuzione dell'intera pipeline di ML con:
if __name__ == "__main__":
    tune_models(RANDOM_FOREST_TRAIN_DATA_PATH, GRADIENT_BOOSTING_TRAIN_DATA_PATH, RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH)

    evaluate_models(RANDOM_FOREST_TEST_DATA_PATH, GRADIENT_BOOSTING_TEST_DATA_PATH, RANDOM_FOREST_MODEL_PATH, GRADIENT_BOOSTING_MODEL_PATH)