#SP23-BAI-001
# KNN ALGORITHM
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Use a relative path (assuming the dataset is in the same directory or a subdirectory)
file_path = './UCI_Credit_Card.csv'  # Replace with the relative path to the dataset
data = pd.read_csv(file_path)

#print(data.isnull().sum()) 

#Handle missing values
imputer = SimpleImputer(strategy='mean')
imputeddata = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)


#Extract Target Values
targetvalues = imputeddata['default.payment.next.month'].values  # targetvalues
targetnames = np.unique(targetvalues)

#Extract DATA Values
dataexcludetarget = imputeddata.drop(columns=['default.payment.next.month']).values #datavalues
featuresnames = list(imputeddata.columns[:-1])

print("Class distribution before SMOTE:")
print(pd.Series(targetvalues).value_counts())

#Class Imbalncing (Balancing)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(dataexcludetarget, targetvalues)

print("Shape Of Dataset after Smote Balancing",X_balanced.shape)


print("Class distribution after SMOTE:")
print(pd.Series(y_balanced).value_counts())

#Feature Scaling (Before Train-Test Split)
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)


#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.2, random_state=42)

knn=KNeighborsClassifier()


#Dictionary for Grid Search Optimizor
param_gridknn={
    'n_neighbors':[3,5,7,9,11],
    'metric': ['euclidean', 'manhattan']
}


grid_search=GridSearchCV(estimator=knn,param_grid=param_gridknn,cv=5)
grid_search.fit(X_train,y_train)

print("Grid Search Best paramaters",grid_search.best_params_)
print("Grid Search Best Score",grid_search.best_score_)


param_dist={
    'n_neighbors':randint(3,20),
    'metric': ['euclidean', 'manhattan']
}

randomsearch=RandomizedSearchCV(estimator=knn, param_distributions=param_dist,n_iter=10,cv=5,random_state=42)
randomsearch.fit(X_train,y_train)

print("Random Search Best Paramaters ",randomsearch.best_params_)
print("Random Search Best Score",randomsearch.best_score_)

bestknn=grid_search.best_estimator_

bestknn.fit(X_train,y_train)

y_prediction = bestknn.predict(X_test)

print("Accuracy score:", accuracy_score(y_test, y_prediction))
print("Precision score:", precision_score(y_test, y_prediction))
print("Recall Score:", recall_score(y_test, y_prediction))
print("F1 Score:", f1_score(y_test, y_prediction))
print("Confusion Matrix:", confusion_matrix(y_test, y_prediction))




# Define performance metrics
metrics = {
    'Accuracy': accuracy_score(y_test, y_prediction),
    'Precision': precision_score(y_test, y_prediction),
    'Recall': recall_score(y_test, y_prediction),
    'F1 Score': f1_score(y_test, y_prediction)
}

# Plot bar chart for performance metrics
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green', 'red'])
plt.title('Performance Metrics', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Metrics', fontsize=14)
plt.ylim(0, 1.1)  # Metrics are between 0 and 1
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=12)
plt.tight_layout()
plt.show()

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_prediction)
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
            xticklabels=targetnames, yticklabels=targetnames)
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.tight_layout()
plt.show()



