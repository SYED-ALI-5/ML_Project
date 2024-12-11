# SP23-BAI-050
# SYED AHMAD ALI
# DATASET_NAME = "Default of Credit Card Clients Dataset"
#Applying Random Forest

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

file_path = './UCI_Credit_Card.csv'
data = pd.read_csv(file_path)

"Pre-Processing Data"

"Information about the credit card DataSet"
print(data)
print(data.info())
print(data.describe())
print(data.columns)

"Calculating the percentage of missing data for each column"

missing_percentage = (data.isnull().sum() / len(data)) * 100
print(missing_percentage)
"No Handling of missing data bcz there is no any missing data in the data set."

"Converting the credit card DataSet into a DATAFRAME"

df = pd.DataFrame(data)
print(df)

"""Checking Class Imbalance Issues:

Imbalanced data means the number of data points across a class label is not uniform.
Output Interpretation:
1- Balanced Dataset: If the counts of each class are roughly equal, your dataset is balanced.
2- Imbalanced Dataset: If one class significantly outnumbers the other, it indicates class imbalance."""

class_distribution = data['default.payment.next.month'].value_counts()
print(f"Original Class Distribution: {class_distribution}")

class_distribution.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

"Splitting features (X) and target (y)"

X = df.drop('default.payment.next.month', axis=1)
y = df['default.payment.next.month']

""" Resolution of the class imbalance distribution issue. """

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X,y)

"Checking the new class distribution"

print("Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())

class_distribution = pd.Series(y_resampled).value_counts()
class_distribution.plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

"Applying Train Test Split of Dataset in 80/20 Ratio"

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

"Scaling"
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parameter Grid for RandomizedSearchCV
paramiterGrid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}

startTime = time.time()
# Initialize Random Forest Classifier
randomForest = RandomForestClassifier()

# RandomizedSearchCV
model = RandomizedSearchCV(
    estimator=randomForest,
    param_distributions=paramiterGrid,
    n_iter=50,  # Reduced to 50 iterations for quicker training
    cv=3,       # 3-fold cross-validation
    verbose=2,  # Enable verbosity
    n_jobs=-1,  # Utilize all processors
    scoring="accuracy"
)

model.fit(X_train, y_train)

print("\nBest Parameters Found: ")
bestParameter = model.best_estimator_

# Predictions
y_pred = bestParameter.predict(X_test)

endTime = time.time()
executionTime = endTime - startTime
hours = int(executionTime // 3600)
minutes = int((executionTime % 3600) // 60)
seconds = executionTime % 60



# Evaluation Metrics
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print(f"Execution Time: {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

# Confusion Matrix Display
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred), display_labels=np.unique(y_test))
disp.plot()
plt.show()

# Visualization of Performance Metrics
classification_report_dict = classification_report(y_test, y_pred, output_dict=True)
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
precision = classification_report_dict['1']['precision']
recall = classification_report_dict['1']['recall']
f1_score = classification_report_dict['1']['f1-score']

metric_values = [precision, recall, f1_score, accuracy]

plt.bar(metrics, metric_values, color=['blue', 'green', 'orange', 'red'])
plt.title('Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.show()