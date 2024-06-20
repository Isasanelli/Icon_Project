import pandas as pd
import logging

# Impostazione del logger per registrare gli eventi
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        logging.info("Dati caricati con successo.")
        return data
    except FileNotFoundError:
        logging.error("File non trovato. Assicurati che il file 'diabetes_data.csv' sia nella directory.")
        raise
    except pd.errors.EmptyDataError:
        logging.error("Il file è vuoto. Fornisci un file CSV con dati.")
        raise