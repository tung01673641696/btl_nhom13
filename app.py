from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

df = pd.read_csv('attachment_default.csv')

df['student'] = df['student'].map({'No': 0, 'Yes': 1})
df['default'] = df['default'].map({'No': 0, 'Yes': 1})
X = df.drop('default', axis=1)
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    

log_model = LogisticRegression(solver='lbfgs')
knn_model = KNeighborsClassifier()
svm_model = SVC()

log_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

def predict_default(model, features):
    probability = model.predict_proba(features)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(features)
    return probability[0]


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

@app.route('/compare_measurements', methods=['POST'])
def compareMeasurements():
    accuracy_log, precision_log, recall_log, f1_log = evaluate_model(log_model, X_test, y_test)
    accuracy_knn, precision_knn, recall_knn, f1_knn = evaluate_model(knn_model, X_test, y_test)
    accuracy_svm, precision_svm, recall_svm, f1_svm = evaluate_model(svm_model, X_test, y_test)

    return jsonify({
        'logistic': {
            'accuracy': accuracy_log,
            'precision': precision_log,
            'recall': recall_log,
            'f1_score': f1_log
        },
        'knn': {
            'accuracy': accuracy_knn,
            'precision': precision_knn,
            'recall': recall_knn,
            'f1_score': f1_knn
        },
        'svm': {
            'accuracy': accuracy_svm,
            'precision': precision_svm,
            'recall': recall_svm,
            'f1_score': f1_svm
        }
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    student = int(data['student'])
    balance = float(data['balance'])
    income = float(data['income'])

    features = np.array([[student, balance, income]])
    features = scaler.transform(features)

    log_prob = predict_default(log_model, features)
    prediction = "Khách hàng có thể hoàn trả tín dụng" if log_prob >= 0.5 else "Khách hàng không thể hoàn trả tín dụng"
    
    return jsonify({
        'prediction': prediction,
        'probability': float(log_prob),
    })

@app.route('/compare', methods=['GET'])
def compare():
    accuracy_log = log_model.score(X_test, y_test)
    accuracy_knn = knn_model.score(X_test, y_test)
    accuracy_svm = svm_model.score(X_test, y_test)
    
    return jsonify({
        'log_accuracy': accuracy_log,
        'knn_accuracy': accuracy_knn,
        'svm_accuracy': accuracy_svm,
    })

if __name__ == '__main__':
    app.run(debug=True)