#K-Means Clustering#
import preprocess
from sklearn.cluster import KMeans
from joblib import dump


def cluster_kmeans(X_train_tfidf):
    # Set n_init explicitly to avoid the FutureWarning
    model_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # n_init=10 to match current default and avoid warning
    model_kmeans.fit(X_train_tfidf)
    # Save the model to a file
    dump(model_kmeans, 'kmeans_model.joblib')
    clusters = model_kmeans.labels_
    print("Cluster labels:", clusters)
    
X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess.preprocess_and_vectorize('dataset.csv')
    
cluster_kmeans(X_train_tfidf)

from sklearn.metrics import silhouette_score

# Example: Finding the optimal number of clusters based on silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    # Set n_init explicitly to avoid the FutureWarning
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_train_tfidf)
    score = silhouette_score(X_train_tfidf, kmeans.labels_)
    silhouette_scores.append((n_clusters, score))

# Find the number of clusters with the highest silhouette score
best_n_clusters, best_score = max(silhouette_scores, key=lambda x: x[1])
print(f"Optimal number of clusters: {best_n_clusters} with a silhouette score of {best_score}")