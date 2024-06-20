import sys
import os
import pandas as pd

# Aggiungere il percorso del modulo alla variabile di percorso di sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.data_loader import load_data
from Code.feature_engineering import feature_engineering
from Code.visualization import visualize_distributions
from Code.model_training import prepare_data, train_and_evaluate_models, train_neural_network, ensure_consistency

def main():
    # Caricamento dei dati
    data = load_data('diabetes_data.csv')

    # Visualizzazione delle informazioni e statistiche
    data.head()
    data.info()
    data.describe()

    # Ingegnerizzazione delle caratteristiche
    data = feature_engineering(data)

    # Preparazione dei dati per l'addestramento
    X_train, X_test, y_train, y_test, scaler, training_columns = prepare_data(data)

    # Addestramento e valutazione dei modelli supervisionati
    train_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Addestramento di un modello MLP (Multi-layer Perceptron)
    mlp_model = train_neural_network(X_train, y_train, X_test, y_test)

    # Esempio di nuovi dati e valutazione con MLP
    new_data = pd.DataFrame({
        'Age': [50],
        'Gender': [1],
        'Ethnicity': [0],
        'SocioeconomicStatus': [1],
        'EducationLevel': [2],
        'BMI': [30],
        'Smoking': [0],
        'AlcoholConsumption': [5],
        'PhysicalActivity': [4],
        'FastingBloodSugar': [120],
        'HbA1c': [6.5],
        'CholesterolTotal': [200],
        'FamilyHistoryDiabetes': [1],
        'PolycysticOvarySyndrome': [0],
        'PreviousPreDiabetes': [1],
        'Hypertension': [0]
    })

    # Ingegnerizzazione delle caratteristiche per i nuovi dati
    new_data = feature_engineering(new_data)

    # Assicurare la consistenza dei nuovi dati
    new_data = ensure_consistency(new_data, training_columns)

    # Standardizzazione dei nuovi dati
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=new_data.columns)

    # Predizione con il modello MLP
    predictions = mlp_model.predict(new_data_scaled)

    # Visualizzazione delle distribuzioni e nuove caratteristiche
    visualize_distributions(data, predictions)

if __name__ == "__main__":
    main()
