from src.Step_1_DataPreparation import DataPreparation
#from src.Step_2_EDA import EDA

output_dirs = {
    'Datapreparation': "./outputs/Datapreparation",
    'EDA': "./outputs/EDA",
    'MissingValuegraph': "./outputs/MissingValuegraph",
    'FeatureSelection': "./outputs/FeatureSelection",
    'ModelTraining': "./outputs/ModelTraining",
    'Models': "./outputs/Models",
    'ModelComparison': "./outputs/ModelComparison"
}

preparation = DataPreparation(
    data_path="./data/raw/insurance.csv",
    output_dirs=output_dirs
)

preparation.run_pipeline()