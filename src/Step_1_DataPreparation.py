# Step_1_DataPreparation.py

# Handles Data Collection, Loading, Understanding, Cleaning,
# Preprocessing, and Saving for the Medical Insurance Cost dataset.

# Dataset Columns:
# age, sex, bmi, children, smoker, region, charges

import os
import pandas as pd
from sklearn.impute import SimpleImputer


class DataPreparation:
    def __init__(self, data_path, output_dirs=None):
        self.data_path = data_path

        if output_dirs is None:
            output_dirs = {
                "Datapreparation": "./outputs/Datapreparation",
                "MissingValuegraph": "./outputs/MissingValuegraph",
            }

        self.output_dirs = output_dirs

        # Populated as the pipeline runs
        self.insurance_df = None
        self.insurance_df_cleaned = None

    # 1. Create Directories
    def create_directories(self):
        for folder in self.output_dirs.values():
            os.makedirs(folder, exist_ok=True)

        print("Directories created successfully.")

    # 2. Data Collection / Loading
    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self.insurance_df = pd.read_csv(self.data_path)

        print("Dataset loaded successfully.")
        print(f"Dataset Shape: {self.insurance_df.shape}")

        return self.insurance_df

    # 3. Understand Data
    def understand_data(self):
        print("\n--- Data Information ---")
        self.insurance_df.info()

        print("\n--- First 5 Rows ---")
        print(self.insurance_df.head())

        print("\n--- Missing Values ---")
        print(self.insurance_df.isnull().sum())

        print("\n--- Summary Statistics ---")
        print(self.insurance_df.describe(include="all"))

    # 4. Clean Data
    def clean_data(self):
        before = len(self.insurance_df)

        # Remove duplicate rows
        self.insurance_df.drop_duplicates(inplace=True)

        after = len(self.insurance_df)
        print(f"Duplicates Removed: {before - after}")

        # Standardize column names
        self.insurance_df.columns = (
            self.insurance_df.columns
            .str.strip()
            .str.lower()
        )

        self.insurance_df_cleaned = self.insurance_df.copy()

        return self.insurance_df_cleaned

    # 5. Handle Missing Values
    def handle_missing_data(self):
        numeric_columns = self.insurance_df_cleaned.select_dtypes(
            include="number"
        ).columns

        missing_before = self.insurance_df_cleaned.isnull().sum().sum()

        imputer = SimpleImputer(strategy="median")

        self.insurance_df_cleaned[numeric_columns] = imputer.fit_transform(
            self.insurance_df_cleaned[numeric_columns]
        )

        missing_after = self.insurance_df_cleaned.isnull().sum().sum()

        print(
            f"Missing Values: {missing_before} before -> {missing_after} after"
        )

        return self.insurance_df_cleaned

    # 6. Save Cleaned Dataset
    def save_cleaned_data(self):
        save_path = os.path.join(
            self.output_dirs["Datapreparation"],
            "insurance_cleaned.csv"
        )

        self.insurance_df_cleaned.to_csv(save_path, index=False)

        print(f"Cleaned dataset saved to: {save_path}")

    # 7. Run Complete Pipeline
    def run_pipeline(self):
        self.create_directories()
        self.load_data()
        self.understand_data()
        self.clean_data()
        self.handle_missing_data()
        self.save_cleaned_data()

        print("\nData Preparation Completed Successfully.")

        return self.insurance_df_cleaned