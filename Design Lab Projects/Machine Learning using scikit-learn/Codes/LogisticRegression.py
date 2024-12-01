#LOGISTIC REGRESSION#
# Logistic Regression Model Training and Evaluation Code
import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import dump

def train_evaluate_lr(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_tfidf, y_train)
    predictions = model_lr.predict(X_test_tfidf)
    dump(model_lr, 'logistic_regression_model.joblib')
    print("Confusion Matrix:", confusion_matrix(y_test, predictions))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess.preprocess_and_vectorize('dataset.csv')


# Call the function with TF-IDF vectors and labels
train_evaluate_lr(X_train_tfidf, y_train, X_test_tfidf, y_test)


#Hyperparameter Tuning#
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define the parameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'newton-cg', 'lbfgs'],
    'max_iter': [1000, 2000, 3000]
}

# Initialize the Logistic Regression model
model_lr = LogisticRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model_lr, param_grid=param_grid, scoring='accuracy', cv=5)

# Fit GridSearchCV
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters found
print("Best Parameters:", grid_search.best_params_)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, predictions))

# Include the best set of hyperparameters in your report
print("Final Best Set of Hyperparameters for Logistic Regression:")
print(grid_search.best_params_)