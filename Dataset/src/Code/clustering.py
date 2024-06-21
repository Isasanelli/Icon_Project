import logging
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def perform_clustering(data):
    from sklearn.cluster import KMeans
    kmeans = KMeans( n_clusters=3, random_state=42 )
    clusters = kmeans.fit_predict( data )
    return clusters


def evaluate_clusters(clusters, data):
    score = silhouette_score( data, clusters )
    logging.info( f'Silhouette Score: {score}' )

    cluster_distribution = {i: sum( clusters == i ) for i in set( clusters )}
    logging.info( f'Cluster Distribution: {cluster_distribution}' )

    plt.figure( figsize=(10, 6) )
    sns.countplot( x=clusters )
    plt.title( 'Cluster Distribution' )
    plt.xlabel( 'Clusters' )
    plt.ylabel( 'Count' )
    plt.show()
