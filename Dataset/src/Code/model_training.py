import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

def plot_metrics(model_metrics):
    models = list(model_metrics.keys())
    accuracies = [model_metrics[model]['accuracy'] for model in models]
    precisions = [model_metrics[model]['precision'] for model in models]
    recalls = [model_metrics[model]['recall'] for model in models]
    f1_scores = [model_metrics[model]['f1'] for model in models]
    roc_aucs = [model_metrics[model]['roc_auc'] for model in models]

    x = range(len(models))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.bar(x, accuracies)
    plt.xticks(x, models, rotation='vertical')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')

    plt.subplot(2, 3, 2)
    plt.bar(x, precisions)
    plt.xticks(x, models, rotation='vertical')
    plt.ylabel('Precision')
    plt.title('Model Precision')

    plt.subplot(2, 3, 3)
    plt.bar(x, recalls)
    plt.xticks(x, models, rotation='vertical')
    plt.ylabel('Recall')
    plt.title('Model Recall')

    plt.subplot(2, 3, 4)
    plt.bar(x, f1_scores)
    plt.xticks(x, models, rotation='vertical')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score')

    plt.subplot(2, 3, 5)
    plt.bar(x, roc_aucs)
    plt.xticks(x, models, rotation='vertical')
    plt.ylabel('ROC AUC')
    plt.title('Model ROC AUC')

    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.title(f'Confusion Matrix for {model_name}')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    logging.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logging.info(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, ROC AUC: {roc_auc:.2f}")

    model_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

    plot_predictions(y_test, y_pred, model_name)

    return model_metrics

def prepare_data(X, y):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_knn(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    return train_and_evaluate_model(model, "K-Nearest Neighbors", X_train, y_train, X_test, y_test)

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier()
    return train_and_evaluate_model(model, "Decision Tree", X_train, y_train, X_test, y_test)

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()
    return train_and_evaluate_model(model, "Random Forest", X_train, y_train, X_test, y_test)

def train_adaboost(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(algorithm='SAMME')
    return train_and_evaluate_model(model, "AdaBoost", X_train, y_train, X_test, y_test)

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier()
    return train_and_evaluate_model(model, "Gradient Boosting", X_train, y_train, X_test, y_test)

def train_naive_bayes(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    return train_and_evaluate_model(model, "Naive Bayes", X_train, y_train, X_test, y_test)
