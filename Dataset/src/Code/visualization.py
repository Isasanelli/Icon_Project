import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import roc_curve, confusion_matrix


# Impostazione del logger per registrare gli eventi
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def plot_histogram(data, feature_name, title, color):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature_name], kde=True, bins=30, color=color)
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel("Frequenza")
    plt.show()

def visualize_distributions(data, predictions):
    old_data = data[data["Diagnosis"] == 0]
    new_data = data[data["Diagnosis"] == 1]

    plot_histogram(old_data, "Age", "Distribuzione dell'Età", color='blue')
    plot_histogram(old_data, "BMI", "Distribuzione del BMI", color='green')
    plot_histogram(old_data, "CholesterolTotal", "Distribuzione del Colesterolo Totale", color='orange')
    plot_histogram(old_data, "HbA1c", "Distribuzione del HbA1c", color='purple')
    plot_histogram(old_data, "FastingBloodSugar", "Distribuzione della Glicemia a Digiuno", color='red')

    plot_histogram(new_data, "Age", "Distribuzione dell'Età (Nuovi Dati)", color='blue')
    plot_histogram(new_data, "BMI", "Distribuzione del BMI (Nuovi Dati)", color='orange')
    plot_histogram(new_data, "CholesterolTotal", "Distribuzione del Colesterolo Totale (Nuovi Dati)", color='green')
    plot_histogram(new_data, "HbA1c", "Distribuzione del HbA1c (Nuovi Dati)", color='purple')
    plot_histogram(new_data, "FastingBloodSugar", "Distribuzione della Glicemia a Digiuno (Nuovi Dati)", color='blue')

def plot_model_performance(model_name, cv_scores, y_test, y_pred, y_proba):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', label='Cross-Validation Scores')
    plt.title(f'Performance del Modello: {model_name}')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(cv_scores) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice di Confusione: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Chance')
    plt.title(f'ROC Curve: {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)
    plt.show()
