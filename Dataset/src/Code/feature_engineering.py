import numpy as np

import logging

def drop_irrelevant_features(data, features_to_drop):
    data = data.drop(columns=features_to_drop, errors='ignore')
    logging.info(f"Eliminate le seguenti feature: {features_to_drop}")
    return data

def feature_engineering(data):
    with np.errstate(divide='ignore', invalid='ignore'):
        if 'Age' in data.columns and 'BMI' in data.columns:
            data["Rapporto_BMI_Età"] = np.where(data["Age"] != 0, data["BMI"] / data["Age"], 0)
        if 'FastingBloodSugar' in data.columns and 'HbA1c' in data.columns:
            data["Differenza_Glicemia_HbA1c"] = data["FastingBloodSugar"] - data["HbA1c"]
        if 'FastingBloodSugar' in data.columns and 'CholesterolTotal' in data.columns:
            data["Rapporto_Glicemia_Colesterolo"] = np.where(data["CholesterolTotal"] != 0, data["FastingBloodSugar"] / data["CholesterolTotal"], 0)

    if 'SocioeconomicStatus' in data.columns and 'EducationLevel' in data.columns:
        data["Indice_Socioeconomico"] = data["SocioeconomicStatus"] + data["EducationLevel"]
    if 'Smoking' in data.columns and 'AlcoholConsumption' in data.columns and 'PhysicalActivity' in data.columns:
        data["Indice_Comportamento_Sociale"] = data["Smoking"] + data["AlcoholConsumption"] + data["PhysicalActivity"]

    return data
