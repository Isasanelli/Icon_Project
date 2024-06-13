import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib

# Caricamento dei dati
data = pd.read_csv('diabetes_data.csv')

# Visualizzazione distribuzione dei dati
plt.figure(figsize=(6, 4))
sns.histplot(data['Age'], kde=True, bins=30)
plt.title('Distribuzione dell\'età')
plt.xlabel('Età')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data['BMI'], kde=True, bins=30)
plt.title('Distribuzione del BMI')
plt.xlabel('BMI')
plt.ylabel('Frequenza')
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data['FastingBloodSugar'], kde=True, bins=30)
plt.title('Distribuzione della glicemia a digiuno')
plt.xlabel('Glicemia a digiuno')
plt.ylabel('Frequenza')
plt.show()

# Eliminazione delle colonne non necessarie
data.drop(columns=['DoctorInCharge', 'PatientID'], inplace=True)

# Analisi della variabile target
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=data)
plt.title('Distribuzione della diagnosi')
plt.xlabel('Diagnosi (0 = No Diabete, 1 = Diabete)')
plt.ylabel('Conteggio')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Diagnosis', y='Age', data=data)
plt.title('Diagnosi vs Età')
plt.xlabel('Diagnosi (0 = No Diabete, 1 = Diabete)')
plt.ylabel('Età')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='Diagnosis', y='BMI', data=data)
plt.title('Diagnosi vs BMI')
plt.xlabel('Diagnosi (0 = No Diabete, 1 = Diabete)')
plt.ylabel('BMI')
plt.show()

# Pairplot delle feature selezionate
selected_features = ['Age', 'BMI', 'FastingBloodSugar', 'HbA1c', 'CholesterolTotal', 'Diagnosis']
sns.pairplot(data[selected_features], hue='Diagnosis')
plt.show()

# Preparazione dei dati
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelli di classificazione
class_models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Addestramento dei modelli di classificazione
for model_name, model in class_models.items():
    model.fit(X_train, y_train)
    print(f"{model_name} trained.")

# Valutazione dei modelli di classificazione
for model_name, model in class_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")

    # Report di classificazione
    print(f"Report di classificazione per {model_name}:\n")
    print(classification_report(y_test, y_pred))

# Modelli di regressione per previsioni
reg_models = {
    'Linear Regression': LinearRegression()
}

# Addestramento dei modelli di regressione
for model_name, model in reg_models.items():
    model.fit(X_train, y_train)
    print(f"{model_name} trained.")

# Valutazione dei modelli di regressione
for model_name, model in reg_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.2f}")

# Funzione principale
def main():
    # Esempio di nuovi dati (sostituire con nuovi dati reali)
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
        # Aggiungere altre colonne necessarie con valori di esempio
    })

    # Assicurati che le feature corrispondano a quelle utilizzate durante l'addestramento
    missing_cols = set(X.columns) - set(nuovi_dati.columns)
    for c in missing_cols:
        nuovi_dati[c] = 0
    nuovi_dati = nuovi_dati[X.columns]

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

    for model_name, model in reg_models.items():
        pred = model.predict(nuovi_dati_scaled)
        previsioni[model_name] = {
            'previsione': pred
        }

    print(previsioni)

if __name__ == "__main__":
    main()
