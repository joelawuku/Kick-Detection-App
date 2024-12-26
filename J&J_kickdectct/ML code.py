import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from swarmlib.pso.particle import Particle


# PSO function
def pso(objective_function, dim, n_particles, n_iterations):
    swarm = [Particle(dim) for _ in range(n_particles)]
    global_best_position = np.zeros(dim)
    global_best_score = float('inf')

    for _ in range(n_iterations):
        for particle in swarm:
            score = objective_function(particle.position)
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()

            # Update velocity
            particle.velocity = 0.5 * particle.velocity + \
                                2 * np.random.random() * (particle.best_position - particle.position) + \
                                2 * np.random.random() * (global_best_position - particle.position)

            # Update position
            particle.position = particle.position + particle.velocity

    pass


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

# Initialize lists to store evaluation metrics
evaluations = []

# Perform 10-fold cross-validation for each model
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for model_name, model in models.items():
    accuracies = []
    precisions = []
    recalls = []
    learning_rate = model.get_params().get('learning_rate', 'N/A')  # Default to 'N/A' if not available

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))

    # Calculate average metrics
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    testing_rate = len(test_index) / len(X)  # Proportion of the data used for testing in each fold

    # Save evaluation metrics to list
    evaluations.append({
        'Model': model_name,
        'Accuracy': mean_accuracy,
        'Precision': mean_precision,
        'Recall': mean_recall,
        'Learning Rate': learning_rate,
        'Testing Rate': testing_rate
    })

# Save evaluation metrics to CSV file
evaluation_df = pd.DataFrame(evaluations)
evaluation_df.to_csv('model_eval.csv', index=False)

