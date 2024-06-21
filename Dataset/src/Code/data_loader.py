import pandas as pd
import logging

def load_data(file_path):
    logging.info("Caricamento dei dati dal file: %s", file_path)
    data = pd.read_csv(file_path)
    logging.info("Dati caricati con successo.")
    return data
