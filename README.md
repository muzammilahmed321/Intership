
Readme · MD
# 🏥 Health Insurance Cost Prediction — Data Science Pipeline
 
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
 
**Research Internship Program 2026 – Batch 2 | Task 1**
**Intern:** Muzammil Ahmed
**GitHub Repository:** https://github.com/muzammilahmed321/Intership
 
---
 
## 📋 1. Project Overview
 
This project implements a complete, end-to-end **Data Science and Machine Learning pipeline** that predicts an individual's **health insurance charges** based on personal and lifestyle attributes — age, BMI, smoking status, region, and more.
 
The pipeline follows a structured, **Object-Oriented Programming (OOP)** architecture, with each stage of the data science workflow implemented as its own dedicated, reusable Python class.
 
| | |
|---|---|
| 🎯 **Problem type** | Regression — predicting a continuous value (insurance charges in USD) |
| 🤖 **Models used** | Random Forest Regressor & Gradient Boosting Regressor |
| 📊 **Dataset size** | 1,338 rows × 7 columns |
 
---
 
## 📁 2. Dataset Information
 
**Source:** Medical Cost Personal Dataset (`insurance.csv`)
 
| Column | Type | Description |
|---|---|---|
| 🎂 `age` | Numeric | Age of the primary policyholder |
| 👤 `sex` | Categorical | Policyholder gender (male / female) |
| ⚖️ `bmi` | Numeric | Body Mass Index |
| 👶 `children` | Numeric | Number of dependents covered |
| 🚬 `smoker` | Categorical | Whether the policyholder smokes (yes / no) |
| 🗺️ `region` | Categorical | Residential region (NE, NW, SE, SW) |
| 💵 `charges` | **Target** | Individual medical insurance costs billed, in USD |
 
> This dataset was selected as it fits the "Quantitative & Qualitative (Financial, Records)" data category, while being clean and well-suited to implementing and documenting the complete pipeline within the project timeline.
 
---
 
## ⚙️ 3. Installation Steps
 
```bash
git clone https://github.com/muzammilahmed321/Intership.git
cd Intership
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```
 
---
 
## 🐍 4. Virtual Environment Setup
 
A dedicated `venv/` virtual environment isolates this project's dependencies from the system Python installation. It is excluded from version control via `.gitignore` and must be created locally using the steps above.
 
---
 
## 📦 5. Dependencies
 
| Package | Purpose |
|---|---|
| `scikit-learn` | Model training, evaluation, cross-validation, hyperparameter tuning |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scipy` | Statistical functions used in EDA |
| `seaborn` / `matplotlib` | Data visualization |
| `joblib` | Trained model serialization (`.pkl` files) |
 
---
 
## 🗂️ 6. Project Structure
 
```
Intership/
├── 📁 data/
│   └── raw/
│       └── insurance.csv
├── 📁 src/
│   ├── Step_1_DataPreparation.py
│   ├── Step_2_EDA.py
│   ├── Step_3_Data_Transformation.py
│   ├── Step_4_FeatureSelection.py
│   ├── Step_5_Train_Test_DataSplit.py
│   ├── Step_6_Model_Training.py
│   ├── Step_7_model_evaluation.py
│   ├── Step_8_Cross_Validation.py
│   ├── Step_9_HyperparameterOptimization.py
│   └── Step_10_Model_Comparison.py
├── 📁 outputs/
│   ├── Datapreparation/        # Cleaned dataset
│   ├── MissingValuegraph/      # Before/after missing-value plots
│   ├── EDA/                    # Distribution, correlation, category plots
│   ├── Transformed/            # Scaled + encoded dataset
│   ├── FeatureSelection/       # Correlation heatmaps, selected features
│   ├── ModelTraining/          # Actual vs predicted scatter plots
│   ├── Models/                 # Saved trained models (.pkl)
│   └── ModelComparison/        # Comparison table and bar charts
├── Main.py                     # Pipeline driver script
├── requirements.txt
├── README.md
└── .gitignore
```
 
---
 
## ▶️ 7. How to Run the Project
 
```bash
venv\Scripts\Activate.ps1
python Main.py
```
 
This executes the full pipeline end-to-end — from raw data loading through final model comparison — printing progress and metrics to the console while saving all visualizations and result tables to `outputs/`.
 
---
 
## 🔄 8. Pipeline Stages
 
| # | Stage | What it does |
|---|---|---|
| 1️⃣ | **Data Preparation** | Loads `insurance.csv`, removes duplicates, standardizes column names, imputes missing values |
| 2️⃣ | **EDA** | Category counts, charges distribution, smoker/region breakdowns, correlation analysis |
| 3️⃣ | **Data Transformation** | Engineers `bmi_category` feature; scales numeric columns; encodes categorical columns |
| 4️⃣ | **Feature Selection** | Computes correlation with target, retains features above a threshold |
| 5️⃣ | **Data Splitting** | 80/20 train/test split |
| 6️⃣ | **Model Training** | Trains Random Forest & Gradient Boosting; saves models as `.pkl` |
| 7️⃣ | **Model Evaluation** | RMSE, MAE, R² Score + Actual vs Predicted plots |
| 8️⃣ | **Cross-Validation** | 5-fold CV for a more reliable performance estimate |
| 9️⃣ | **Hyperparameter Optimization** | `GridSearchCV` tuning of `n_estimators` / `max_depth` |
| 🔟 | **Model Comparison** | Side-by-side metrics table + bar charts |
 
---
 
## 📈 9. Results
 
> ⚠️ **Note:** Replace the placeholders below with the actual values printed after running `python Main.py`.
 
| 🤖 Model | 📉 RMSE | 📊 MAE | 📈 R² Score |
|---|---|---|---|
| 🌲 Random Forest | *[4816.09]* | *[2737.60]* | *[0.8738]* |
| 🚀 Gradient Boosting | *[4761.96]* | *[2678.67]* | *[0.8766]* |
 
**🔁 Cross-Validation (Random Forest):** Mean RMSE = *[5080.23]*, Std = *[ 171.34]*
 
**🏆 Best Hyperparameters (Random Forest):** *[fill in from `best_params` output]*
 
**💡 Key Finding:** Across exploratory analysis, smoking status was consistently the strongest predictor of insurance charges, with smokers incurring substantially higher costs than non-smokers regardless of other factors. *[Update once you review your actual outputs.]*
 
---
 
## 🧩 10. Challenges Faced
 
- 🔧 The original reference pipeline was designed for a three-file, group-based dataset. Significant restructuring was needed to adapt it to a single-file, per-record dataset — removing merge, region-cleaning, and plant-pruning logic with no equivalent here.
- 🐛 A data type bug surfaced during feature engineering: a BMI-based category feature created with `pd.cut()` defaulted to a `category` dtype, silently skipped by the categorical encoder. Fixed by explicitly casting to `str` before encoding.
- 🗃️ Git required cleanup after `venv/` and `__pycache__` were briefly tracked before `.gitignore` was properly configured — resolved with `git rm -r --cached`.
---
 
## 🚀 11. Future Improvements
 
- ✨ Add a dedicated Prediction/Inference script for generating predictions on new, unseen records
- 🎛️ Extend hyperparameter tuning to the Gradient Boosting model
- 🧪 Benchmark against additional models (XGBoost, LightGBM)
- ✅ Add automated unit tests to the CI/CD pipeline
---
 
<p align="center">
  Made with 🧠 and ☕ as part of the Research Internship Program 2026
</p>
