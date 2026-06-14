# Regression-Family-Comparison
Comparison of Linear Regression, Polynomial Regression, Ridge, Lasso, Elastic Net and Logistic Regression using Insurance Dataset.

# Regression Family Comparison Project using Machine Learning

## Project Overview

This project compares multiple regression algorithms on an Insurance dataset and evaluates their predictive performance using regression metrics.

The project also includes Logistic Regression as a classification benchmark and compares its performance against transformed regression predictions.

The objective is to identify the best-performing regression model through parameter tuning and systematic evaluation.

---

## Dataset

Insurance Dataset

### Regression Target

* Charges

### Classification Target

* Smoker Status

  * Yes = 1
  * No = 0

---

## Data Preprocessing

### One-Hot Encoding

Categorical features encoded:

* Sex
* Region
* Smoker

### Feature Selection

Selected features:

* Sex
* Region
* BMI
* Children

### Train-Test Split

The dataset was divided into:

* Training Set: 80%
* Testing Set: 20%

---

## Machine Learning Algorithms Used

### Regression Family

1. Linear Regression
2. Polynomial Regression
3. Ridge Regression
4. Lasso Regression
5. Elastic Net Regression

### Classification

6. Logistic Regression

---

## Hyperparameter Optimization

### Polynomial Regression

Tuned Parameter:

* Degree = [1, 2, 3, 4, 5]

Best degree selected using R² Score.

### Ridge Regression

Tuned Parameter:

* Alpha = [0.01, 0.1, 1, 2, 3, 4]

### Lasso Regression

Tuned Parameter:

* Alpha = [0.01, 0.1, 1, 2, 3, 4]

### Elastic Net Regression

Tuned Parameters:

* Alpha = [0.01, 0.1, 1, 2, 3, 4]
* L1 Ratio = [0.2, 0.5, 0.8]

The best parameter combination was automatically selected based on model performance.

---

## Evaluation Metrics

### Regression Metrics

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## Project Workflow

Insurance Dataset

↓

Data Encoding

↓

Feature Selection

↓

Train-Test Split

↓

Linear Regression

↓

Polynomial Regression

↓

Ridge Regression

↓

Lasso Regression

↓

Elastic Net Regression

↓

Logistic Regression

↓

Performance Evaluation

↓

Best Model Selection

↓

Final Comparison Report

---

## Results

The project automatically:

* Finds the best Polynomial Degree
* Finds the best Ridge Alpha
* Finds the best Lasso Alpha
* Finds the best Elastic Net Alpha
* Finds the best Elastic Net Ratio
* Evaluates all regression models
* Selects the best-performing model
* Compares regression performance against Logistic Regression

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn

---

## Skills Demonstrated

* Regression Analysis
* Regularization Techniques
* Hyperparameter Tuning
* Feature Engineering
* Classification Metrics
* Model Evaluation
* Machine Learning Pipelines
* Comparative Model Analysis

---

## Key Learning Outcomes

Through this project, I learned:

* How Linear Regression works
* How Polynomial Regression captures non-linear patterns
* The effect of Ridge Regularization
* The effect of Lasso Regularization
* How Elastic Net combines Ridge and Lasso
* How to evaluate regression models using MAE, MSE, RMSE, and R²
* How to build machine learning pipelines
* How to compare multiple machine learning algorithms systematically
* How to select the best-performing model based on evaluation metrics

