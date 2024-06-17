import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import logging

# Configurazione del logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(
            "File not found. Ensure the file 'diabetes_data.csv' is in the directory."
        )
        raise
    except pd.errors.EmptyDataError:
        logging.error("The file is empty. Provide a CSV file with data.")
        raise


def visualize_data(data):
    """Visualize dataset with various plots."""
    plt.figure(figsize=(6, 4))
    sns.histplot(data["Age"], kde=True, bins=30)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(data["BMI"], kde=True, bins=30)
    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.histplot(data["FastingBloodSugar"], kde=True, bins=30)
    plt.title("Fasting Blood Sugar Distribution")
    plt.xlabel("Fasting Blood Sugar")
    plt.ylabel("Frequency")
    plt.show()

    selected_features = [
        "Age",
        "BMI",
        "FastingBloodSugar",
        "HbA1c",
        "CholesterolTotal",
        "Diagnosis",
    ]
    sns.pairplot(data[selected_features], hue="Diagnosis")
    plt.show()


def feature_engineering(data):
    """Create new features based on existing ones."""
    data["BMI_Age_Ratio"] = data["BMI"] / data["Age"]
    data["Sugar_HbA1c_Difference"] = data["FastingBloodSugar"] - data["HbA1c"]
    data["Sugar_Cholesterol_Ratio"] = (
        data["FastingBloodSugar"] / data["CholesterolTotal"]
    )
    data["BMI_Adjusted"] = data["BMI"] / data["Age"]
    data["Sugar_HbA1c_Ratio"] = data["FastingBloodSugar"] / data["HbA1c"]
    data["Socioeconomic_Index"] = data["SocioeconomicStatus"] + data["EducationLevel"]
    data["Social_Behavior_Index"] = (
        data["Smoking"] + data["AlcoholConsumption"] + data["PhysicalActivity"]
    )

    return data


def visualize_new_features(data):
    """Visualize the new features created during feature engineering."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data["BMI_Age_Ratio"], kde=True, bins=30)
    plt.title("Distribuzione del rapporto BMI/Age")
    plt.xlabel("BMI/Age")
    plt.ylabel("Frequenza")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["Sugar_HbA1c_Difference"], kde=True, bins=30)
    plt.title("Distribuzione della differenza FastingBloodSugar - HbA1c")
    plt.xlabel("Differenza FastingBloodSugar - HbA1c")
    plt.ylabel("Frequenza")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(data["Sugar_Cholesterol_Ratio"], kde=True, bins=30)
    plt.title("Distribuzione del rapporto FastingBloodSugar / CholesterolTotal")
    plt.xlabel("FastingBloodSugar / CholesterolTotal")
    plt.ylabel("Frequenza")
    plt.show()


def prepare_data(data):
    """Prepare the dataset for training."""
    X = data.drop(
        columns=["Diagnosis", "DoctorInCharge", "PatientID"]
    )  # Exclude non-numeric and irrelevant columns
    y = data["Diagnosis"]
    training_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, training_columns


def train_and_evaluate_models(X, y):
    """Train and evaluate different classification models using K-Fold cross-validation."""
    class_models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    # Inizializzazione K-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for model_name, model in class_models.items():
        accuracies = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
        precisions = cross_val_score(model, X, y, cv=kf, scoring="precision")
        recalls = cross_val_score(model, X, y, cv=kf, scoring="recall")
        f1_scores = cross_val_score(model, X, y, cv=kf, scoring="f1")
        roc_aucs = cross_val_score(model, X, y, cv=kf, scoring="roc_auc")

        results.append(
            {
                "Model": model_name,
                "Accuracy Mean": accuracies.mean(),
                "Accuracy Std": accuracies.std(),
                "Precision Mean": precisions.mean(),
                "Precision Std": precisions.std(),
                "Recall Mean": recalls.mean(),
                "Recall Std": recalls.std(),
                "F1 Score Mean": f1_scores.mean(),
                "F1 Score Std": f1_scores.std(),
                "ROC AUC Mean": roc_aucs.mean(),
                "ROC AUC Std": roc_aucs.std(),
            }
        )

        model.fit(X, y)

    results_df = pd.DataFrame(results)
    return results_df, class_models


def ensure_consistency(input_data, training_columns):
    """Ensure the new data has the same features as the training data."""
    missing_cols = set(training_columns) - set(input_data.columns)
    extra_cols = set(input_data.columns) - set(training_columns)

    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data.drop(columns=extra_cols)
    input_data = input_data[training_columns]

    return input_data


def make_predictions(new_data, scaler, models, training_columns):
    """Make predictions on new data using trained models."""
    new_data_scaled = pd.DataFrame(scaler.transform(new_data), columns=training_columns)
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(new_data_scaled)
        pred_proba = model.predict_proba(new_data_scaled)[:, 1]
        predictions[model_name] = {"prediction": pred, "probability": pred_proba}
    return predictions


def main():
    data = load_data("diabetes_data.csv")
    visualize_data(data)

    data = feature_engineering(data)
    visualize_new_features(data)

    X_train, X_test, y_train, y_test, scaler, training_columns = prepare_data(data)
    results_df, models = train_and_evaluate_models(X_train, y_train)

    # Print the results in a formatted table
    logging.info(
        "Model Evaluation Results (K-Fold Cross Validation):\n%s",
        results_df.to_string(index=False),
    )

    new_data = pd.DataFrame(
        {
            "Age": [50],
            "Gender": [1],
            "Ethnicity": [0],
            "SocioeconomicStatus": [1],
            "EducationLevel": [2],
            "BMI": [30],
            "Smoking": [0],
            "AlcoholConsumption": [5],
            "PhysicalActivity": [4],
            "FastingBloodSugar": [120],
            "HbA1c": [6.5],
            "CholesterolTotal": [200],
            "FamilyHistoryDiabetes": [1],
            "GestationalDiabetes": [0],
            "PolycysticOvarySyndrome": [0],
            "PreviousPreDiabetes": [1],
            "Hypertension": [0],
        }
    )

    new_data = feature_engineering(new_data)
    new_data = ensure_consistency(new_data, training_columns)
    predictions = make_predictions(new_data, scaler, models, training_columns)

    for model_name, pred in predictions.items():
        logging.info(
            f"{model_name} Prediction: {pred['prediction'][0]}, Probability: {pred['probability'][0]:.2f}"
        )


if __name__ == "__main__":
    main()
