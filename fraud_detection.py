# Import necessary libraries
import zipfile
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Upload any ZIP file
print("Please upload your dataset (ZIP file)")
uploaded = files.upload()  # This allows any file to be uploaded

# Get the uploaded filename dynamically
uploaded_filename = list(uploaded.keys())[0]  # This captures whatever file the user uploads

# Step 2: Extract and Load the Data
with zipfile.ZipFile(uploaded_filename, 'r') as zip_ref:
    zip_ref.extractall('extracted_files')

# List extracted files
extracted_files = os.listdir('extracted_files')
print("Extracted files:", extracted_files)

# Automatically detect CSV file (so it works for different datasets)
csv_file = None
for file in extracted_files:
    if file.endswith('.csv'):
        csv_file = f'extracted_files/{file}'
        break

if not csv_file:
    raise FileNotFoundError("No CSV file found in the uploaded ZIP.")

# Load the dataset
df = pd.read_csv(csv_file)

# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Show first few rows
print("\nFirst few rows of the dataset:")
print(df.head())

# Step 3: Feature and target separation
X = df.drop('Class', axis=1)  # Features (all columns except 'Class')
y = df['Class']  # Target (the 'Class' column)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Handling class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 6: Feature scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Step 7: Train the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Step 8: Model evaluation
y_pred = model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy
accuracy = (y_pred == y_test).mean()
print(f"\nAccuracy: {accuracy:.4f}")
