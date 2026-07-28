from src.Step_1_DataPreparation import DataPreparation
from src.Step_2_EDA import EDA
from src.Step_3_Data_Transformation import DataTransform
from src.Step_4_FeatureSelection import FeatureSelection
from src.Step_5_Train_Test_DataSplit import DataSplitter
from src.Step_6_Model_Training import ModelTrainer
from src.Step_7_model_evaluation import ModelEvaluator
from src.Step_8_Cross_Validation import CrossValidation
from src.Step_9_HyperparameterOptimization import HyperparameterTuner
from src.Step_10_Model_Comparison import ModelComparator
from sklearn.ensemble import RandomForestRegressor

output_dirs = {
    'Datapreparation': "./outputs/Datapreparation",
    'MissingValuegraph': "./outputs/MissingValuegraph",
    'EDA': "./outputs/EDA"
}

# Step 1 Data Preparation
preparation = DataPreparation(
    data_path="./data/raw/insurance.csv",
    output_dirs=output_dirs
)
preparation.run_pipeline()

# Step 2 EDA
eda = EDA(
    insurance_df_filled=preparation.insurance_df_cleaned,
    output_dir=output_dirs['EDA']
)
eda.run_eda()

# Step 3 Data Transformation
transformer = DataTransform(
    insurance_df_cleaned=preparation.insurance_df_cleaned,
    output_dir="./outputs/Transformed"
)
transformed_df = transformer.run_transformation()

# Step 4 Feature Selection
selector = FeatureSelection(
    transformed_df=transformed_df,
    output_dir="./outputs/FeatureSelection"
)
selected_df, selected_features = selector.run_feature_selection(threshold=0.05)

print("Feature selection complete.")
print("Selected features:", selected_features)
print("Final dataset shape for modeling:", selected_df.shape)

# Step 5 Data Splitting
splitter = DataSplitter(selected_df=selected_df, target_col='charges')
X_train, X_test, y_train, y_test = splitter.run_split()

# Step 6 Model Training
trainer = ModelTrainer(X_train=X_train, y_train=y_train)
rf_model = trainer.train_model('rf')
gbr_model = trainer.train_model('gbr')

# Step 7 Model Evaluation
rf_eval = ModelEvaluator(rf_model, X_test, y_test, 'RandomForest')
rf_metrics, rf_pred = rf_eval.evaluate()

gbr_eval = ModelEvaluator(gbr_model, X_test, y_test, 'GradientBoosting')
gbr_metrics, gbr_pred = gbr_eval.evaluate()

# Step 8 Cross Validation
cv = CrossValidation(
    RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, y_train, 'RandomForest'
)
cv_mean, cv_std = cv.run_cv()

# Step 9 Hyperparameter Optimization
tuner = HyperparameterTuner(
    RandomForestRegressor(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [None, 10]},
    X_train, y_train, 'RandomForest', cv_folds=3
)
best_model, best_params = tuner.run_grid_search()

# Step 10 Model Comparison
comparator = ModelComparator()
comparison_df = comparator.compare([rf_metrics, gbr_metrics])

print("Pipeline complete.")
print("Best RandomForest hyperparameters found:", best_params)
print("Final Model Comparison:")
print(comparison_df)