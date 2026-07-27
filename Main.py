from src.Step_1_DataPreparation import DataPreparation
from src.Step_2_EDA import EDA
from src.Step_3_Data_Transformation import DataTransform
from src.Step_4_FeatureSelection import FeatureSelection
output_dirs = {
    'Datapreparation': "./outputs/Datapreparation",
    'MissingValuegraph': "./outputs/MissingValuegraph",
    'EDA': "./outputs/EDA"
}

preparation = DataPreparation(
    data_path="./data/raw/insurance.csv",
    output_dirs=output_dirs
)
preparation.run_pipeline()

eda = EDA(
    insurance_df_filled=preparation.insurance_df_cleaned,   # ← right side matches DataPreparation's actual attribute
    output_dir=output_dirs['EDA']
)
eda.run_eda()

transformer = DataTransform(
    insurance_df_cleaned=preparation.insurance_df_cleaned,
    output_dir="./outputs/Transformed"
)
transformed_df = transformer.run_transformation()

selector = FeatureSelection(
    transformed_df=transformed_df,
    output_dir="./outputs/FeatureSelection"
)
selected_df, selected_features = selector.run_feature_selection(threshold=0.05)

print("\nFeature selection complete.")
print("Selected features:", selected_features)
print("Final dataset shape for modeling:", selected_df.shape)