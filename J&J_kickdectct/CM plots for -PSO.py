import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Load the dataset
data = pd.read_csv('Openlab_wellA.csv')

# Preprocessing - Normalizing the dataset
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['Class']))
y = data['Class']

# Define machine learning models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Perform 10-fold cross-validation for each model
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for model_name, model in models.items():
    cms = np.zeros((2, 2))  # Initialize confusion matrix

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cms += confusion_matrix(y_test, y_pred)

    # Convert confusion matrix to integers
    cms = cms.astype(int)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cms, annot=True, fmt='d', cmap='Reds', cbar=False, annot_kws={"size": 16})
    plt.title(f'Confusion Matrix for {model_name}-PSO', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.xticks([0.5, 1.5], ['NO KICK', 'KICK'], fontsize=12)
    plt.yticks([0.5, 1.5], ['NO KICK', 'KICK'], fontsize=12)
    plt.show()
