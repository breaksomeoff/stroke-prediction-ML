"""
run_pipeline.py

Script per orchestrare l'intero processo progettuale:
1. Esecuzione dell'EDA (data_exploration.py)
2. Preprocessing e bilanciamento (data_preprocessing.py)
3. Addestramento del modello e validazione (model.py)
4. Valutazione finale (evaluation.py)

Ogni fase viene eseguita come script separato per mantenere la modularità.
"""

import os
import sys
import subprocess
import logging

# Configuriamo il logging di base
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def run_script(script_path):
    """
    Esegue lo script specificato e gestisce eventuali errori.
    In questo caso non catturiamo l'output, in modo che i log dei processi figli
    vengano stampati direttamente sullo standard output.
    """
    logging.info(f"Esecuzione dello script: {script_path}")
    # Eseguiamo lo script senza catturare l'output per lasciare che i log del child process
    # vengano inviati direttamente a stdout/stderr.
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        logging.error(f"Errore nell'esecuzione di {script_path}. Codice di ritorno: {result.returncode}")
        sys.exit(result.returncode)
    else:
        logging.info(f"Script {script_path} completato con successo.")

def main():
    try:
        # Impostiamo il percorso base per gli script nella cartella "src"
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

        # Definizione dei percorsi degli script
        eda_script = os.path.join(base_dir, "data_exploration.py")
        preprocessing_script = os.path.join(base_dir, "data_preprocessing.py")
        model_script = os.path.join(base_dir, "model.py")
        evaluation_script = os.path.join(base_dir, "evaluation.py")

        # 1. Esecuzione dell'EDA
        run_script(eda_script)

        # 2. Preprocessing e bilanciamento
        run_script(preprocessing_script)

        # 3. Addestramento del modello
        run_script(model_script)

        # 4. Valutazione finale
        run_script(evaluation_script)

        logging.info("[MAIN] Pipeline completata con successo!")
    except Exception as e:
        logging.error(f"[MAIN] Errore nell'esecuzione della pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
