import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score

# Load Dataset
data = pd.read_csv("./UCI_Credit_Card.csv")
y = data['default.payment.next.month']
data.drop("default.payment.next.month", axis=1, inplace=True)

# Define Hyperparameter Grid
param_grid = {
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear', 'saga'],               
    'max_iter': [100, 200, 300]
}

# Preprocessing: Standardize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Dimensionality Reduction using PCA
pca = PCA(n_components=8)  
X_pca = pca.fit_transform(X_scaled)

# Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=42)
X_pca, y = smote.fit_resample(X_pca, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Logistic Regression Initialization
lr = LogisticRegression()

# Hyperparameter Optimization using RandomizedSearchCV
random_search_cv_lr = RandomizedSearchCV(
    estimator=lr, 
    param_distributions=param_grid, 
    cv=5, 
    verbose=0, 
    n_jobs=2
)

# Training the Model and Measuring Execution Time
start_time = time.time()
random_search_cv_lr.fit(X_train, y_train)
end_time = time.time()

# Best Estimator and Prediction
best_estimators_lr = random_search_cv_lr.best_estimator_ 
print(best_estimators_lr)
predictions_lr = best_estimators_lr.predict(X_test)

# Performance Metrics
execution_time_lr = end_time - start_time
accuracy_lr = accuracy_score(y_test, predictions_lr)
precision_lr = precision_score(y_test, predictions_lr)
recall_lr = recall_score(y_test, predictions_lr)
f1_lr = f1_score(y_test, predictions_lr)
cm_lr = confusion_matrix(y_test, predictions_lr)
cR = classification_report(y_test, predictions_lr)

# Display Results
print(f"Classification Matrix:\n{cR}")
print(f"Confusion Matrix:\n{cm_lr}")
print(f"Accuracy: {accuracy_lr:.2f}")
print(f"Precision: {precision_lr:.2f}")
print(f"Recall: {recall_lr:.2f}")
print(f"F1-Score: {f1_lr:.2f}")
print(f"Execution Time: {execution_time_lr:.2f} seconds")

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=np.unique(y_test))
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Bar Chart for Performance Metrics
classification_report_dict = classification_report(y_test, predictions_lr, output_dict=True)
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
metric_values = [precision_lr, recall_lr, f1_lr, accuracy_lr]

plt.bar(metrics, metric_values, color=['orange', 'blue', 'green', 'red'])
plt.title('Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()
