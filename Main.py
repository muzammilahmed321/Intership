from src.Step_1_DataPreparation import DataPreparation
from src.Step_2_EDA import EDA
# ...and so on for all 11

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
    demand_path="./data/raw/demand.csv",
    plants_path="./data/raw/plants.csv",
    generation_path="./data/raw/generation_costs.csv",
    output_dirs=output_dirs
)