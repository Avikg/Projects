#K-Nearest Neighbors (KNN)#
import preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Suppress specific warnings related to KNeighborsClassifier
warnings.filterwarnings("ignore", message="cannot use tree with sparse input: using brute force")

def train_evaluate_knn(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train_tfidf, y_train)
    predictions = model_knn.predict(X_test_tfidf)
    dump(model_knn, 'knn_model.joblib')
    # Evaluation metrics
    print("KNN Confusion Matrix:", confusion_matrix(y_test, predictions))
    print("KNN Accuracy:", accuracy_score(y_test, predictions))
    print("KNN Precision:", precision_score(y_test, predictions))
    print("KNN Recall:", recall_score(y_test, predictions))
    print("KNN F1 Score:", f1_score(y_test, predictions))
    
X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess.preprocess_and_vectorize('dataset.csv')

train_evaluate_knn(X_train_tfidf, y_train, X_test_tfidf, y_test)

# Define parameter grid
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# Initialize the model
model_knn = KNeighborsClassifier()

# Initialize GridSearchCV
grid_search_knn = GridSearchCV(estimator=model_knn, param_grid=param_grid_knn, cv=5, scoring='accuracy')

# Fit to the data
grid_search_knn.fit(X_train_tfidf, y_train)

# Best parameters
print("Best parameters for KNN:", grid_search_knn.best_params_)