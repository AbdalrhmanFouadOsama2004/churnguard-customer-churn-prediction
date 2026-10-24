# ChurnGuard — Customer Churn Prediction

> **An end-to-end machine learning project for predicting customer churn and understanding the factors associated with customer attrition.**

ChurnGuard analyzes the Telco Customer Churn dataset, prepares customer and service features, compares multiple machine-learning and deep-learning models, and evaluates their ability to identify customers who may leave a service.

## Project Overview

Customer churn can affect recurring revenue, customer lifetime value, and long-term growth. This project explores how historical customer information can be used to estimate churn risk and support data-informed retention strategies.

The work is implemented in a Jupyter Notebook: `Customer_Churn_Prediction.ipynb`.

## Objectives

- Explore the dataset and investigate churn patterns.
- Clean data and prepare numerical and categorical features.
- Engineer additional features that describe tenure, charges, and service usage.
- Address class imbalance in the training data using SMOTE.
- Train and compare several classification approaches.
- Evaluate results with metrics suited to binary classification.
- Examine feature importance and validate model performance.

## Dataset

The notebook downloads the **Telco Customer Churn** dataset using `kagglehub` from the Kaggle dataset reference `blastchar/telco-customer-churn`.

The data includes customer demographics, subscribed services, contract and payment details, tenure, charges, and the target column `Churn`.

- **Target:** `Churn` (`Yes` / `No`), mapped to binary values (`1` / `0`).
- **Customer identifier:** `customerID`, removed during cleaning.
- **Data cleaning:** `TotalCharges` is converted to numeric; invalid/blank values are filled with the column median.

Please review the dataset provider's terms and attribution requirements before redistributing the data.

## Workflow

1. **Load data** — download and read the Telco Customer Churn CSV.
2. **Exploratory Data Analysis (EDA)** — inspect missing values, summary statistics, target distribution, categorical churn rates, numerical distributions, and correlations.
3. **Clean data** — convert `TotalCharges`, handle missing values, remove the customer identifier, and encode the target.
4. **Feature engineering** — create tenure groups, charge-related ratios/averages, and a service-count feature.
5. **Preprocess** — one-hot encode categorical variables, split the data into training and test sets using stratification, scale features, and apply SMOTE to the training set.
6. **Train models** — compare Logistic Regression, Random Forest, XGBoost, Gradient Boosting, and a neural network.
7. **Evaluate** — compare Accuracy, Precision, Recall, F1-score, and ROC-AUC; inspect confusion matrices, classification reports, ROC curves, and Precision–Recall curves.
8. **Interpret and validate** — review feature importance for tree-based models and run cross-validation for the selected non-neural model where applicable.

## Models

| Model | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted trees |
| Gradient Boosting | Boosted decision-tree ensemble |
| Deep Learning | Feed-forward neural network with dense layers, batch normalization, dropout, and regularization |

The notebook selects the best model by **F1-score** and also identifies the best model by **ROC-AUC**. The winning model and its scores should be taken from the results produced when the notebook is run; no performance figures are stated here because they depend on the execution environment and outputs.

## Evaluation Metrics

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted churners who actually churn.
- **Recall:** proportion of actual churners identified by the model.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** ability to rank churn and non-churn cases across classification thresholds.
- **Confusion matrix and classification report:** show the types and counts of classification outcomes.

For churn use cases, accuracy should not be interpreted alone. Recall, precision, F1-score, and the cost of false positives versus false negatives are important when choosing a model and decision threshold.

## Tech Stack

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib, Seaborn, and Plotly
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- TensorFlow / Keras
- KaggleHub

## Getting Started

### 1. Clone or download the project

Place `Customer_Churn_Prediction.ipynb` in your project directory.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

Run in a terminal:

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost tensorflow plotly jupyter
```

### 4. Run the notebook

```bash
jupyter notebook Customer_Churn_Prediction.ipynb
```

Run the cells from top to bottom. The notebook downloads the dataset through KaggleHub, so an internet connection is required for the initial download. Deep-learning training time depends on your hardware.

## Project Structure

```text
ChurnGuard/
├── Customer_Churn_Prediction.ipynb
└── README.md
```

## Important Methodological Notes

- SMOTE is applied to the **training split**, not the held-out test set, which helps keep test evaluation representative of the original distribution.
- The current notebook scales features before applying SMOTE and trains several models on the resampled, scaled training data.
- For a production-grade experiment, consider placing preprocessing and resampling inside an `imblearn` pipeline and fitting that pipeline independently within each cross-validation fold. This helps prevent information leakage during validation.
- Tree-based models generally do not require feature scaling, so compare appropriately configured pipelines if optimizing each model.
- The engineered feature `AvgMonthlyCharges` uses `TotalCharges / (tenure + 1)` in the notebook; interpret it as the notebook's engineered feature rather than a standard billing measure.
- Model performance is an estimate on this dataset, not a guarantee of future customer behavior. Any real deployment should include monitoring, privacy safeguards, and periodic retraining.

## Limitations and Future Improvements

- Tune hyperparameters with a leakage-safe cross-validation workflow.
- Compare threshold selection strategies based on retention costs.
- Add calibration analysis for predicted probabilities.
- Explain individual predictions with SHAP or another interpretability method.
- Package preprocessing and the selected model into a reproducible inference pipeline.
- Build a simple dashboard or API for batch and single-customer predictions.
- Monitor drift and performance after deployment.

## Intended Use

This project is intended for learning, experimentation, and portfolio demonstration. Predictions should support—not replace—business judgment, and should not be used as the sole basis for consequential decisions about individual customers.
## Author

Abdalrhman Attia, https://www.linkedin.com/in/abdalrhman-fouad-osama-attia-06147025a/, https://github.com/AbdalrhmanFouadOsama2004.
---

**Suggested repository name:** `churnguard-customer-churn-prediction`

**Suggested short description:** `Customer churn prediction with EDA, feature engineering, SMOTE, and machine-learning/deep-learning model comparison.`
