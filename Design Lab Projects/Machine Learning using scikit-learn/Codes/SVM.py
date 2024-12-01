#Support Vector Machine (SVM)#
import preprocess
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_evaluate_svm(X_train_tfidf, y_train, X_test_tfidf, y_test):
    model_svm = SVC(kernel='linear')
    model_svm.fit(X_train_tfidf, y_train)
    predictions = model_svm.predict(X_test_tfidf)
    dump(model_svm, 'svm_model.joblib')
    # Evaluation metrics
    print("SVM Confusion Matrix:", confusion_matrix(y_test, predictions))
    print("SVM Accuracy:", accuracy_score(y_test, predictions))
    print("SVM Precision:", precision_score(y_test, predictions))
    print("SVM Recall:", recall_score(y_test, predictions))
    print("SVM F1 Score:", f1_score(y_test, predictions))
    
X_train_tfidf, X_test_tfidf, y_train, y_test = preprocess.preprocess_and_vectorize('dataset.csv')

train_evaluate_svm(X_train_tfidf, y_train, X_test_tfidf, y_test)


from sklearn.model_selection import GridSearchCV

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

model_svm = SVC()
grid_search_svm = GridSearchCV(model_svm, param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train_tfidf, y_train)
print("Best parameters for SVM:", grid_search_svm.best_params_)