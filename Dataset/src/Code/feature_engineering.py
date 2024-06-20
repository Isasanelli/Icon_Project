import numpy as np

def feature_engineering(data):
    with np.errstate(divide='ignore', invalid='ignore'):
        data["Rapporto_BMI_Età"] = np.where(data["Age"] != 0, data["BMI"] / data["Age"], 0)
        data["Differenza_Glicemia_HbA1c"] = data["FastingBloodSugar"] - data["HbA1c"]
        data["Rapporto_Glicemia_Colesterolo"] = np.where(data["CholesterolTotal"] != 0, data["FastingBloodSugar"] / data["CholesterolTotal"], 0)

    data["Indice_Socioeconomico"] = data["SocioeconomicStatus"] + data["EducationLevel"]
    data["Indice_Comportamento_Sociale"] = data["Smoking"] + data["AlcoholConsumption"] + data["PhysicalActivity"]

    return data
