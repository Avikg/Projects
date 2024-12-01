#Neural Networks#
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import preprocess

# Assume preprocess_and_vectorize function returns the necessary split data
X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess.preprocess_and_vectorize('dataset.csv')

# Scikit-learn's MLPClassifier works directly with sparse matrices, no need to convert them to dense
def train_evaluate_mlp(X_train, y_train, X_test, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("MLP Accuracy:", accuracy)
    # Save the model
    # Joblib is typically used for saving scikit-learn models due to compatibility
    from joblib import dump
    dump(mlp, 'mlp_model.joblib')

train_evaluate_mlp(X_train_tfidf, y_train, X_test_tfidf, y_test)

# If you still want to perform hyperparameter tuning similar to what you were doing with keras_tuner,
# you can use GridSearchCV or RandomizedSearchCV from Scikit-learn.
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define the parameter space for RandomizedSearch
param_distributions = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (200,)],
    'alpha': uniform(0.0001, 0.001),  # Regularization term
}

# Create an MLPClassifier instance to use as the estimator
mlp = MLPClassifier(max_iter=200)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(mlp, param_distributions=param_distributions, n_iter=5, scoring='accuracy', n_jobs=-1, cv=3)

# Perform the search on a part of the data to save time
X_train_partial, X_test_partial, y_train_partial, y_test_partial = train_test_split(X_train_tfidf, y_train, test_size=0.5, random_state=42)

random_search.fit(X_train_partial, y_train_partial)

# Print the best parameters and accuracy
print("Best parameters found: ", random_search.best_params_)
print("Best accuracy found: ", random_search.best_score_)
