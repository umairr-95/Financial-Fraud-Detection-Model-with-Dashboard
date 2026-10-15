# Financial Fraud Detection System Using Machine Learning

A machine learning project that analyzes financial transaction data, compares classification algorithms, and provides an interactive Streamlit dashboard for fraud analysis and prediction.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Using the Dashboard](#using-the-dashboard)
- [Model Artifacts](#model-artifacts)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Author](#author)

## Project Overview

Financial fraud is a major challenge in digital payments and financial transactions. Machine learning can help identify suspicious patterns in transaction records and support fraud investigation.

This project develops a Financial Fraud Detection System using Python and scikit-learn. It follows a machine learning workflow that includes data cleaning, exploratory data analysis (EDA), feature engineering, model training, performance evaluation, and deployment through a Streamlit dashboard.

The dashboard allows users to explore transaction data, review fraud-related patterns, compare model results when available, and submit transaction details for a prediction.

**Project type:** Machine Learning, Data Analysis, and Interactive Dashboard

**Application:** Academic prototype for financial transaction fraud analysis

## Objectives

The main objectives of this project are:

- Clean and prepare financial transaction data for analysis.
- Explore patterns in genuine and fraudulent transactions.
- Create additional features from transaction dates and other available information.
- Train and compare multiple machine learning classification algorithms.
- Evaluate models using appropriate classification metrics.
- Save the trained model for reuse.
- Build an interactive dashboard using Streamlit.
- Demonstrate how a trained model can support transaction-level fraud prediction.

## Features

- **Data Cleaning:** Inspect missing values, duplicate records, data types, and dataset structure.
- **Exploratory Data Analysis:** Analyze fraud distribution and transaction characteristics.
- **Feature Engineering:** Create time-based and transaction-related features.
- **Machine Learning:** Train and compare three classification algorithms.
- **Model Evaluation:** Review accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- **Interactive Dashboard:** Explore transaction information through a web-based interface.
- **Fraud Prediction:** Enter transaction details and obtain a model prediction.
- **Fraud Analytics:** Explore fraud-related summaries and patterns.
- **Graph Analytics:** Demonstrate relationship-based analysis where supported by the available data.
- **Monitoring and Forecasting Demonstrations:** Explore historical data through demonstration modules; these are not connected to a live banking feed.

## Technology Stack

| Technology | Purpose |
| Python | Main programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Data preprocessing, machine learning, and evaluation |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive dashboard |
| NetworkX | Graph and relationship analysis |
| Jupyter Notebook | Step-by-step experimentation and model development |
| Git and GitHub | Version control and project hosting |

## Project Architecture

The project follows this general workflow:

1. Load the transaction dataset.
2. Inspect and clean the data.
3. Perform exploratory data analysis.
4. Engineer useful features.
5. Separate input features from the target variable.
6. Split the dataset into training and testing sets.
7. Preprocess numerical and categorical features.
8. Train and compare machine learning models.
9. Evaluate model performance.
10. Save the selected model and comparison results.
11. Load the saved artifacts in the Streamlit dashboard.
12. Display analytics and generate predictions from submitted transaction details.

Financial Transaction Dataset
             |
             v
        Data Cleaning
             |
             v
 Exploratory Data Analysis
             |
             v
    Feature Engineering
             |
             v
 Data Preprocessing and Split
             |
             v
       Model Training
             |
             v
    Model Evaluation
             |
             v
 Save Model and Metrics
             |
             v
   Streamlit Dashboard
             |
       +-----+-----+
       |           |
       v           v
   Analytics   Predictions
```

## Project Structure

The expected project structure is shown below. Adjust the folder names to match your actual VS Code workspace.

Fraud Detection model project 2/
│
├── README.md
├── app.py
│
├── models/
│   ├── best_fraud_model.joblib
│   ├── all_fraud_models.joblib
│   └── model_comparison.csv
│
├── Raw_dataset/
│   └── financial_fraud_detection_feature_engineered.csv
│
└── notebooks/
    ├── datacleaning.ipynb
    ├── EDA.ipynb
    ├── FeatureEngineering.ipynb
    └── ModelTraining.ipynb
```

**Note:** The dataset may instead be stored in a `data/` directory. Keep the paths in `app.py` consistent with your actual folder structure. Do not create empty files simply to match this example.

## Dataset Description

The project uses a financial transaction dataset containing transaction details and a target column named `Fraudulent`.

The dataset used during development contains **5,000 transaction records**, with the following class distribution:

| Transaction class | Count | Percentage |
| Genuine (0) | 4,518 | 90.36% |
| Fraudulent (1) | 482 | 9.64% |
| Total | 5,000 | 100% |

The target variable is:

- `Fraudulent = 0`: Genuine transaction.
- `Fraudulent = 1`: Fraudulent transaction.

Available fields include transaction amount, merchant category, payment method, device type, location, international transaction status, previous transaction count, average spending, account age, and other transaction-related attributes.

Engineered features include:

- `Transaction_Hour`
- `Day_of_Week`
- `Transaction_Month`
- `Is_Weekend`
- `High_Amount`
- `Is_Night`

The exact fields depend on the version of the dataset used by the notebooks.

### Data Privacy

If the dataset contains personal, confidential, or licensed information, do not publish it to GitHub without authorization. Use an appropriate public or anonymized dataset when sharing the project.

## Machine Learning Workflow

### 1. Data Cleaning

Notebook: `datacleaning.ipynb`

The data preparation stage inspects the dataset structure, data types, missing values, duplicate records, and other data-quality issues.

The goal is to understand the quality of the source data before performing analysis and model training.

### 2. Exploratory Data Analysis

Notebook: `EDA.ipynb`

EDA is used to understand the dataset through statistical summaries and visualizations.

The analysis can include:

- Genuine versus fraudulent transaction distribution.
- Transaction amount distributions.
- Merchant category patterns.
- Payment method and device distributions.
- Location and international transaction patterns.
- Transaction activity by hour, weekday, or month.

The fraud class is less frequent than the genuine class. Therefore, accuracy alone may not provide a complete picture of model performance.

### 3. Feature Engineering

Notebook: `FeatureEngineering.ipynb`

Feature engineering creates additional variables that may help the model identify useful patterns.

Examples include extracting the transaction hour, day of the week, and month from a transaction timestamp and creating binary indicators for weekend or night-time activity.

The purpose is to represent information in a form that can be used by the machine learning pipeline.

### 4. Data Preprocessing

Notebook: `ModelTraining.ipynb`

The target column is separated from the input features. Identifier columns and the original timestamp may be excluded when suitable derived features are available.

The workflow uses a training and testing split. The preprocessing pipeline handles numerical and categorical columns separately.

Typical preprocessing operations include:

- Median imputation for numerical values.
- Most-frequent imputation for categorical values.
- Standard scaling of numerical features.
- One-hot encoding of categorical features.
- A scikit-learn pipeline to connect preprocessing and model training.

### 5. Model Training

Three classification algorithms are compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The models are trained using the prepared training dataset and evaluated on held-out test data.

### 6. Model Selection

Model comparison helps identify a candidate model based on the evaluation results. The development workflow uses F1-score as a selection criterion.

The final choice should also consider the practical consequences of missed fraud cases and false fraud alerts.

### 7. Model Saving

The selected model is saved using Joblib so that the Streamlit application can load and reuse the fitted pipeline without retraining it every time.

## Models Used

| Model | Description |
| Logistic Regression | A classification baseline that models class probabilities using a linear decision function. |
| Decision Tree | A tree-based classifier that learns decision rules from input features. |
| Random Forest | An ensemble classifier that combines multiple decision trees. |

The models are compared using the same general evaluation workflow. Actual performance depends on the dataset, feature preparation, model configuration, and decision threshold.

## Evaluation Metrics

The project evaluates classification models using the following metrics:

- **Accuracy:** The proportion of all transactions classified correctly.
- **Precision:** The proportion of predicted fraud cases that are actually fraudulent.
- **Recall:** The proportion of actual fraud cases detected by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well the model ranks positive cases above negative cases across classification thresholds.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.

Since the dataset is imbalanced, precision and recall are particularly important when assessing fraud detection performance.

**Important:** Add the actual metric values from your executed `ModelTraining.ipynb` or `model_comparison.csv` before publishing performance claims. Do not report unverified accuracy or other scores.

## Installation

### Prerequisites

Install the following before running the project:

- Python
- Visual Studio Code
- The Python extension for VS Code
- The Jupyter extension if you want to run the notebooks

Open a terminal in the project directory.

### 1. Create a virtual environment (recommended)

python -m venv .venv


Activate it on Windows Command Prompt:

.venv\Scripts\activate


If you use PowerShell and activation is blocked by your execution policy, you can use Command Prompt or invoke the environment's Python executable directly.

### 2. Install dependencies

python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy matplotlib joblib networkx scikit-learn


If you use a Jupyter notebook environment, install Jupyter as well:

python -m pip install notebook ipykernel


## How to Run the Project

### Step 1: Open the project

Open the project folder in VS Code and select the Python interpreter from your virtual environment.

### Step 2: Check the required files

Confirm that the following are available:

- `app.py`
- The feature-engineered CSV dataset
- `models/best_fraud_model.joblib`
- Any optional model comparison file used by the dashboard

The app will not be able to make model predictions if its required model file is missing or incompatible.

### Step 3: Start the Streamlit dashboard

Run this command in the VS Code terminal from the project directory:

python -m streamlit run app.py


Streamlit will start a local server and provide a URL, commonly:

```text
http://localhost:8501
```

Open the displayed URL in your browser.

### Step 4: Explore the dashboard

Use the dashboard navigation to inspect available sections, review transaction analytics, compare models when the results file is available, and test the prediction interface.

If an error occurs, check the terminal output and confirm that the dataset and model paths match the files in your project.

## Using the Dashboard

The application includes the following sections, depending on the available data and implementation:

1. **Executive Dashboard:** Summary metrics and high-level transaction information.
2. **Dataset Overview:** Dataset structure and descriptive information.
3. **EDA Dashboard:** Visual exploration of transaction patterns.
4. **Feature Engineering:** Overview of engineered features.
5. **Fraud Intelligence:** Fraud-related historical summaries.
6. **Model Comparison:** Comparison of saved model evaluation results.
7. **Fraud Prediction:** Transaction-level model prediction.
8. **Fraud Analytics:** Additional fraud-related breakdowns.
9. **Explainable AI:** Model or feature insights supported by the implementation.
10. **Real-Time Monitoring:** A monitoring-style demonstration using available historical data.
11. **Fraud Forecasting:** A historical trend or forecasting demonstration.
12. **Graph Analytics:** Relationship-oriented analysis using graph techniques.

The monitoring and forecasting sections should not be interpreted as validated real-time fraud detection or production forecasting unless those capabilities have been separately implemented and tested.

## Model Artifacts

| File | Purpose |
| `best_fraud_model.joblib` | Saved candidate model or fitted preprocessing-and-model pipeline. |
| `all_fraud_models.joblib` | Saved collection of trained models, if generated by the training notebook. |
| `model_comparison.csv` | Saved model comparison results, if generated. |
| `financial_fraud_detection_feature_engineered.csv` | Dataset used by the dashboard or downstream workflow. |

Keep model artifacts and datasets in the locations expected by your code.

## Limitations

- Model predictions depend on the quality and representativeness of the training data.
- The dataset has an imbalanced class distribution.
- False positives and false negatives are possible.
- Performance metrics on a test dataset do not guarantee performance on future transactions.
- Features must be available at the time of prediction and must not leak information about the target.
- The `Suspicious_Keyword` feature should be audited to ensure it does not reveal the fraud label or use information unavailable at prediction time.
- Monitoring and forecasting demonstrations are not equivalent to a live financial transaction system.
- The project is an academic prototype and is not intended to replace professional fraud investigation or a production banking security platform.

## Future Improvements

Potential extensions include:

- Validate the model using additional datasets and temporal holdout testing.
- Tune classification thresholds based on the costs of missed fraud and false alerts.
- Add automated tests for data preprocessing and prediction inputs.
- Improve model explanations using validated interpretability methods.
- Connect to a secure transaction stream for real-time monitoring.
- Evaluate forecasting methods using suitable time-series validation.
- Add authentication, logging, deployment monitoring, and security controls before any real-world deployment.
- Track model performance and data drift over time.

## Author

**Umair Akbar Mohammed**

Project: Financial Fraud Detection System Using Machine Learning

Technologies: Python, Pandas, Scikit-learn, Matplotlib, Joblib, Streamlit, and NetworkX.

