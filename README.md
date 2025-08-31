# Parkinson's Disease Prediction using Vocal Measurements

## 1. Project Overview

- This project aims to predict the presence of Parkinson's disease in individuals based on a range of vocal measurements. The dataset used contains various biomedical voice measurements from healthy individuals and people with Parkinson's.

- A machine learning pipeline was developed that includes feature selection, data preprocessing, and model training to achieve high accuracy. The final model is deployed in an interactive web application built with Streamlit, allowing users to input voice feature data and receive a real-time prediction.

## Live App Link : https://parkinnson-disease-prediction-07.streamlit.app/

## 2. The Dataset

- Source: Parkinsson disease.csv

- Description: This dataset is composed of 195 voice recordings from 31 individuals, 23 of whom have Parkinson's disease. Each row corresponds to one of the voice recordings, and the columns contain specific voice measures.

- Target Variable: status - 1 for Parkinson's, 0 for healthy.

- Key Features: The model identified that features related to vocal pitch instability and variation, such as PPE (Pitch Period Entropy) and spread1, are highly predictive.

## 3. Machine Learning Workflow

- The project follows a robust machine learning pipeline to ensure the model is both accurate and reliable:

- Data Exploration: Initial analysis to understand feature distributions and correlations.

- Feature Selection: Used Feature Importance

- Handling Class Imbalance: The dataset is imbalanced (more samples with Parkinson's). This was addressed using the SMOTE (Synthetic Minority Over-sampling Technique) on the training data to prevent model bias.

- Model Training & Tuning: An XGBoost (Extreme Gradient Boosting) classifier was chosen. Its hyperparameters were tuned using GridSearchCV with 10-fold cross-validation to find the optimal settings.

- Pipeline Construction: All steps (scaling, SMOTE, and the classifier) were encapsulated in a scikit-learn Pipeline to prevent data leakage and streamline the process.

- Model Evaluation: The final model's performance was evaluated on an unseen test set using metrics like Accuracy, Precision, Recall, and F1-Score.
<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/6e00e0fe-0eec-4139-b071-a780600b2e21" />



# 4. Technology Stack

| Category                  | Tools / Libraries                              |
| :------------------------ | :----------------------------------------------|
| **Programming Language**  | `Python 3.x`                                   |
| **Data Manipulation**     | `pandas`                                       |
| **Machine Learning**      | `scikit-learn`, `xgboost`, `imbalanced-learn`  |
| **Model Persistence**     | `pickle`                                       |
| **Web Framework**         | `streamlit`                                    |
| **Data Visualization**    | `seaborn`, `matplotlib`                        |



## 5. File Structure
```
├── Parkinsson disease.csv      
├── cleaned_parkinsons_data.csv
├── model.ipynb
|__ visualization.ipynb
|__ EDA.ipynb   
├── app.py                     
├── parkinsons_model.pkl        
├── selected_features.pkl      
└── README.md                   
```
## 7. How to Run This Project

Follow these steps to set up and run the prediction application on your local machine.

### Step 1: Clone the Repository & Setup Environment
```
# Clone this repository 
git clone []
cd parkinsons-prediction
```
```
# Create and activate a Conda environment
conda create --name parkinsons_env python=3.9
conda activate parkinsons_env
```

### Step 2: Install Dependencies

Install all the required libraries using pip.
```
pip install pandas scikit-learn imbalanced-learn xgboost streamlit joblib seaborn matplotlib
```

### Step 3: Train the Model

Before you can run the app, you need to train the model and create the .pkl files. Run this script once from your terminal:


### Step 4: Launch the Streamlit App

Now, run the Streamlit app from your terminal:
```
streamlit run app.py
```
Your default web browser will open with the interactive application, ready for you to input data and make predictions.

## 7. Model Performance
The final tuned XGBoost model achieved a robust and reliable performance, with a cross-validated accuracy of approximately 90% on the training set and similar high performance on the final held-out test set, demonstrating good generalization.

### Accuracy : 90%

## 8. Conclusion

This project successfully demonstrates the potential of using machine learning on vocal measurements for the early detection of Parkinson's disease. By following a rigorous workflow including automated feature selection and robust cross-validation, a high-accuracy XGBoost model was developed.


### Author
### Thiyanesh D
