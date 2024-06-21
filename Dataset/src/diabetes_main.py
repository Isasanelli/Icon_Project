import logging
from Code.data_loader import load_data
from Code.feature_engineering import drop_irrelevant_features, feature_engineering
from Code.model_training import prepare_data, train_knn, train_decision_tree, train_random_forest, train_adaboost, \
    train_gradient_boosting, train_naive_bayes, plot_metrics
from Code.clustering import perform_clustering, evaluate_clusters
from Code.knowledge_Base import create_knowledge_base, query_knowledge_base
from Code.data_understanding import explore_data


def main():
    logging.basicConfig( level=logging.INFO )

    # Carica i dati
    data = load_data( 'diabetes_data.csv' )
    explore_data( data )

    # Prepara i dati
    features_to_drop = ['PatientID', 'DoctorInCharge']
    data = drop_irrelevant_features( data, features_to_drop )
    data = feature_engineering( data )

    X = data.drop( 'Diagnosis', axis=1 )
    y = data['Diagnosis']

    X_train, X_test, y_train, y_test = prepare_data( X, y )

    # Addestra i modelli
    model_metrics = {}
    model_metrics['K-Nearest Neighbors'] = train_knn( X_train, y_train, X_test, y_test )
    model_metrics['Decision Tree'] = train_decision_tree( X_train, y_train, X_test, y_test )
    model_metrics['Random Forest'] = train_random_forest( X_train, y_train, X_test, y_test )
    model_metrics['AdaBoost'] = train_adaboost( X_train, y_train, X_test, y_test )
    model_metrics['Gradient Boosting'] = train_gradient_boosting( X_train, y_train, X_test, y_test )
    model_metrics['Naive Bayes'] = train_naive_bayes( X_train, y_train, X_test, y_test )

    # Plot metrics
    plot_metrics( model_metrics )

    # Clustering
    clusters = perform_clustering( X )
    evaluate_clusters( clusters, X )

    # Knowledge Base
    g = create_knowledge_base( data )
    query_knowledge_base( g )


if __name__ == "__main__":
    main()
