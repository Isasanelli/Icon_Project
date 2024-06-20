import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
import logging
import sys
import os

# Aggiungere il percorso del modulo alla variabile di percorso di sistema
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Code.visualization import plot_model_performance

# Impostazione del logger per registrare gli eventi
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def prepare_data(data):
    X = data.drop(columns=["Diagnosis", "DoctorInCharge", "PatientID"])
    y = data["Diagnosis"]
    training_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, training_columns

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    class_models = {
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Support Vector Machine": SVC(probability=True)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in class_models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        logging.info(f"Cross-Validation Scores for {model_name}: {cv_scores}")
        logging.info(f"Mean CV Score for {model_name}: {mean_cv_score:.2f}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])

        logging.info(
            f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}"
        )
        logging.info(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")

        plot_model_performance(model_name, cv_scores, y_test, y_pred, y_proba)


def train_neural_network(X_train, y_train, X_test, y_test):
    mlp_model = MLPClassifier( hidden_layer_sizes=(50, 50), max_iter=500, random_state=42, learning_rate_init=0.001,
                               solver='adam' )  # Modificato il solver e aggiunto learning_rate_init

    # Aggiungi punteggi di cross-validazione
    kf = KFold( n_splits=5, shuffle=True, random_state=42 )
    cv_scores = cross_val_score( mlp_model, X_train, y_train, cv=kf, scoring='accuracy' )

    mlp_model.fit( X_train, y_train )

    y_pred = mlp_model.predict( X_test )
    y_proba = mlp_model.predict_proba( X_test )
    accuracy = accuracy_score( y_test, y_pred )
    precision = precision_score( y_test, y_pred )
    recall = recall_score( y_test, y_pred )
    f1 = f1_score( y_test, y_pred )
    roc_auc = roc_auc_score( y_test, y_proba[:, 1] )

    logging.info(
        f"MLP Classifier - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}"
    )
    logging.info( f"Classification Report for MLP Classifier:\n{classification_report( y_test, y_pred )}" )

    plot_model_performance( "MLP Classifier", cv_scores, y_test, y_pred, y_proba )

    return mlp_model


def ensure_consistency(input_data, training_columns):
    missing_cols = set(training_columns) - set(input_data.columns)
    extra_cols = set(input_data.columns) - set(training_columns)

    for col in missing_cols:
        input_data[col] = 0

    input_data = input_data.drop(columns=extra_cols)
    input_data = input_data[training_columns]

    return input_data
