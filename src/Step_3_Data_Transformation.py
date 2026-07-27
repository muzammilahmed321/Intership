import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataTransform:
    def __init__(self, insurance_df_cleaned, output_dir="./TransformedData"):
        """
        insurance_df_cleaned : cleaned dataframe
        output_dir : folder to save transformed dataset
        """
        self.insurance_df_cleaned = insurance_df_cleaned.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.transformed_df = None
        self.numeric_cols = None
        self.cat_cols = None

    # Scale numeric features and encode categorical features
    def transform_data(self, exclude_col=None):

        if exclude_col is None:
            exclude_col = ["charges"]

        df = self.insurance_df_cleaned.copy()

        # Numeric columns
        self.numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.numeric_cols = [
            col for col in self.numeric_cols if col not in exclude_col
        ]

        scaler = StandardScaler()
        df[self.numeric_cols] = scaler.fit_transform(df[self.numeric_cols])

        # Categorical columns
        self.cat_cols = df.select_dtypes(include="object").columns.tolist()

        for col in self.cat_cols:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])

        self.transformed_df = df

        self.transformed_df.to_csv(
            os.path.join(self.output_dir, "insurance_df_transformed.csv"),
            index=False
        )

        print(f"Transformed DataFrame saved with shape: {self.transformed_df.shape}")

        return self.transformed_df

    # Feature Engineering
    def engineer_features(self):
        """
        Adds a BMI category feature using original BMI values.
        """

        df = self.insurance_df_cleaned.copy()

        bins = [0, 18.5, 25, 30, 100]
        labels = ["underweight", "normal", "overweight", "obese"]

        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=bins,
            labels=labels
        ).astype(str)

        self.insurance_df_cleaned = df

        print("Feature engineered: bmi_category added.")

        return self.insurance_df_cleaned

    # Complete pipeline
    def run_transformation(self):

        self.engineer_features()

        transformed_df = self.transform_data()

        print("\nFull Data Transformation completed successfully.")

        return transformed_df