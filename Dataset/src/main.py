import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Caricamento dei dati
try:
    data = pd.read_csv('diabetes_data.csv')
except FileNotFoundError:
    print("File non trovato. Assicurati che il file 'diabetes_data.csv' sia nella directory.")
    exit()
except pd.errors.EmptyDataError:
    print("Il file è vuoto. Fornisci un file CSV con i dati.")
    exit()

# Visualizzazione del dataset
print(data.head())
print(data.info())
print(data.describe())
print(data.shape)

# Verifica della presenza di valori mancanti
print(data.isnull().sum())

# Impostazioni di stile per i grafici
sns.set(style="whitegrid", font_scale=1.2)

# Visualizzazione della distribuzione delle feature
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Distribuzione dell\'Età')
plt.xlabel('Età')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['BMI'], kde=True, bins=30)
plt.title('Distribuzione dell\'Indice di Massa Corporea (BMI)')
plt.xlabel('BMI')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['FastingBloodSugar'], kde=True, bins=30)
plt.title('Distribuzione della Glicemia a Digiuno')
plt.xlabel('Glicemia a Digiuno')
plt.ylabel('Frequenza')
plt.show()

# Selezione delle feature per il pairplot
selected_features = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c', 'CholesterolTotal', 'Diagnosis']

# Pairplot delle feature selezionate
sns.pairplot(data[selected_features], hue='Diagnosis')
plt.suptitle('Pairplot delle Feature Selezione', y=1.02)
plt.show()

# Feature Engineering con rapporti tra colonne
data['Rapporto_BMI_Età'] = data['BMI'] / data['Age']
data['Differenza_Glicemia_HbA1c'] = data['FastingBloodSugar'] - data['HbA1c']
data['Rapporto_Glicemia_Colesterolo'] = data['FastingBloodSugar'] / data['CholesterolTotal']

# Visualizzazione distribuzione delle nuove feature
plt.figure(figsize=(10, 6))
sns.histplot(data['Rapporto_BMI_Età'], kde=True, bins=30)
plt.title('Distribuzione del Rapporto BMI/Età')
plt.xlabel('Rapporto BMI/Età')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Differenza_Glicemia_HbA1c'], kde=True, bins=30)
plt.title('Distribuzione della Differenza Glicemia - HbA1c')
plt.xlabel('Differenza Glicemia - HbA1c')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Rapporto_Glicemia_Colesterolo'], kde=True, bins=30)
plt.title('Distribuzione del Rapporto Glicemia/Colesterolo')
plt.xlabel('Rapporto Glicemia/Colesterolo')
plt.ylabel('Frequenza')
plt.show()

# Preparazione dei dati per il training del modello
X = data.drop(columns=['Diagnosis', 'DoctorInCharge', 'PatientID'])  # Escludi colonne non numeriche e non rilevanti
y = data['Diagnosis']

# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split dei dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelli di classificazione
class_models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}

# Addestramento e valutazione dei modelli di classificazione
for model_name, model in class_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    if model_name == 'Naive Bayes':
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test).max(axis=1))

    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")
    print(f"Report di classificazione per {model_name}:\n{classification_report(y_test, y_pred)}")

# Funzione per garantire la consistenza delle feature
def ensure_consistency(input_data: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    missing_cols = set(training_columns) - set(input_data.columns)
    extra_cols = set(input_data.columns) - set(training_columns)

    # Aggiungere colonne mancanti con valore 0
    for col in missing_cols:
        input_data[col] = 0

    # Rimuovere colonne extra
    input_data = input_data.drop(columns=extra_cols)

    # Riordinare le colonne per corrispondere all'ordine delle colonne di addestramento
    input_data = input_data[training_columns]

    return input_data

# Funzione principale
def main():
    # Esempio di nuovi dati (da sostituire con dati reali)
    nuovi_dati = pd.DataFrame({
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
        'GestationalDiabetes': [0],
        'PolycysticOvarySyndrome': [0],
        'PreviousPreDiabetes': [1],
        'Hypertension': [0]
    })

    # Feature Engineering sui nuovi dati
    nuovi_dati['Rapporto_BMI_Età'] = nuovi_dati['BMI'] / nuovi_dati['Age']
    nuovi_dati['Differenza_Glicemia_HbA1c'] = nuovi_dati['FastingBloodSugar'] - nuovi_dati['HbA1c']
    nuovi_dati['Rapporto_Glicemia_Colesterolo'] = nuovi_dati['FastingBloodSugar'] / nuovi_dati['CholesterolTotal']

    # Garantire la consistenza delle feature
    nuovi_dati = ensure_consistency(nuovi_dati, X.columns)

    # Predizioni sui nuovi dati
    nuovi_dati_scaled = pd.DataFrame(scaler.transform(nuovi_dati), columns=X.columns)
    previsioni = {}
    for model_name, model in class_models.items():
        pred = model.predict(nuovi_dati_scaled)
        pred_proba = model.predict_proba(nuovi_dati_scaled)[:, 1]
        previsioni[model_name] = {
            'previsione': pred,
            'probabilità': pred_proba
        }

    print(previsioni)

if __name__ == "__main__":
    main()
