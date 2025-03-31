**AI-Powered Credit Card Fraud Detection**

## Overview
This project is an AI-powered fraud detection system that identifies fraudulent credit card transactions using machine learning. The dataset used for this project contains real-world transactions labeled as legitimate or fraudulent. By training a model on these transactions, we can predict fraudulent activities with high accuracy.

## Dataset
The dataset consists of anonymized credit card transactions, including:
- **Time**: Time elapsed since the first transaction
- **V1-V28**: Principal components derived from PCA (to protect confidentiality)
- **Amount**: Transaction amount
- **Class**: Target variable (0 = Legitimate, 1 = Fraudulent)

## Implementation Steps
### **1. Data Upload & Extraction**
- Users upload a ZIP file containing the dataset.
- The system automatically extracts and loads the CSV file inside the ZIP.

### **2. Data Preprocessing**
- Handles missing values (if any)
- Splits data into **features (X)** and **target (y)**
- Splits dataset into **training (80%)** and **testing (20%)** sets
- Addresses class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)**
- Standardizes the feature set for better model performance

### **3. Model Training & Evaluation**
- **Random Forest Classifier** is used to train the model
- Model is evaluated using:
  - **Classification Report** (Precision, Recall, F1-score)
  - **Confusion Matrix** (Visual representation of predictions)
  - **Accuracy Score**

## Results
- The model successfully detects fraudulent transactions with high accuracy.
- **Confusion Matrix** helps visualize misclassifications.
- Precision and recall values indicate the model’s effectiveness in identifying fraud.

## Key Features
✔ **Flexible File Upload:** Users can upload any dataset in ZIP format.  
✔ **Automated Preprocessing:** The script detects and processes the CSV file automatically.  
✔ **Handles Imbalanced Data:** Uses SMOTE to improve fraud detection accuracy.  
✔ **Scalable Model:** Trained using Random Forest, but easily adaptable for other algorithms.  
✔ **Visual Insights:** Confusion matrix provides a clear performance evaluation.  

## How to Use
1. Run the script in Google Colab or a Jupyter Notebook.
2. Upload a ZIP file containing the dataset.
3. Let the model train and evaluate fraud detection.
4. Analyze results from the confusion matrix and classification report.

## Future Enhancements
- Experiment with **deep learning models** like LSTMs.
- Implement **real-time fraud detection** using streaming data.
- Enhance interpretability with **SHAP values** to understand model decisions.

---

This AI-powered fraud detection system provides a strong foundation for identifying fraudulent transactions and can be further refined for real-world applications. 🚀

