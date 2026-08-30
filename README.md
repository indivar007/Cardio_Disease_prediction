# 🫀 Heart Disease Prediction System

## 📌 Project Overview
This project aims to predict the likelihood of a patient having heart disease based on various medical attributes such as age, sex, chest pain type, blood pressure, etc. We have implemented multiple Machine Learning algorithms and optimized them using Hyperparameter Tuning to achieve the best possible accuracy.

## 🛠️ Tech Stack
* Language: Python
* Libraries: Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn
* Techniques: Data Preprocessing, Exploratory Data Analysis (EDA), Hyperparameter Tuning (GridSearchCV)

## 📂 Dataset
The dataset used for this project contains medical details of patients.
* Source: [ UCI Heart Disease Dataset / Kaggle]
*Attributes: Age, Sex, CP (Chest Pain), Trestbps (Resting BP), Chol (Cholesterol), FBS, RestECG, Thalach, Exang, Oldpeak, Slope, CA, Thal, Target.

## 🤖 Machine Learning Models Used
We trained and evaluated the following models:

1.  Logistic Regression: Used as a baseline model for binary classification.
2.  Support Vector Machine (SVM): Tested with different kernels (Linear, Poly, RBF, Sigmoid).
3.  Decision Tree Classifier: Tuned parameters like `max_depth` and `min_samples_split`.
4.  Random Forest Classifier: An ensemble method tuned with `n_estimators` for better robustness.

## ⚙️ Hyperparameter Tuning
To improve the model performance, we used GridSearchCV for:
* Finding the best Solver for Logistic Regression.
* Selecting the optimal Kernel for SVM.
* Tuning Max Depth & Leaf Nodes for Decision Trees.
* Optimizing N_Estimators for Random Forest.

## 📊 Model Performance (Results)
After training and tuning, here are the accuracy scores achieved:


Logistic Regression -> 85.86%
Support Vector Machine (SVM) -> 84.22% 
Decision Tree ->85.56%
Random Forest -> 87%

> Conclusion: The Random forest performed the best with an accuracy of 87%
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the notebook/script:
    ```bash
    jupyter notebook
    # OR
    python main.py
    ```

 🔮 Future Scope
 Deploying the model using a web interface (Streamlit or Flask).
 Trying Deep Learning models (Neural Networks) for better accuracy.
 Collecting more real-time data to improve robustness.

