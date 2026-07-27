import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureSelection:
    def __init__(self, transformed_df, target_col="charges",
                 output_dir="./FeatureSelection"):
        """
        transformed_df : DataFrame after transformation
        target_col : target variable
        output_dir : folder to save outputs
        """

        self.transformed_df = transformed_df.copy()
        self.target_col = target_col
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.corr_target = None
        self.selected_features = None


    # 1. Correlation Heatmap

    def corr_heatmap(self):

        numeric_df = self.transformed_df.select_dtypes(include="number")

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Matrix of Insurance Features", fontsize=16)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "full_correlation_heatmap.png"
            )
        )

        plt.show()
        plt.close()


    # 2. Correlation with Target

    def compute_correlation(self):

        if self.target_col not in self.transformed_df.columns:
            raise ValueError(f"{self.target_col} not found in DataFrame.")

        corrmat = self.transformed_df.corr(numeric_only=True)

        self.corr_target = corrmat[self.target_col].abs()

        self.corr_target.to_csv(
            os.path.join(
                self.output_dir,
                "correlation_with_target.csv"
            )
        )

        plt.figure(figsize=(10, 6))

        cor_sorted = self.corr_target.sort_values(
            ascending=False
        ).dropna()

        cor_sorted.plot(kind="bar")

        plt.title(f"Feature Correlation with {self.target_col}")
        plt.xlabel("Features")
        plt.ylabel("Absolute Correlation")
        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                "correlation_with_target_graph.png"
            )
        )

        plt.show()
        plt.close()

        return self.corr_target


    # 3. Feature Selection

    def select_features(self, threshold=0.05):

        if self.corr_target is None:
            raise ValueError(
                "Run compute_correlation() first."
            )

        filtered = self.corr_target.drop(self.target_col)

        self.selected_features = (
            filtered[filtered > threshold]
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        if len(self.selected_features) == 0:
            print("No features found above the threshold.")
            return pd.DataFrame()

        selected_summary = pd.DataFrame({
            "Feature": self.selected_features,
            "Correlation_with_Target":
                filtered[self.selected_features].values
        })

        selected_summary.to_csv(
            os.path.join(
                self.output_dir,
                "selected_features.csv"
            ),
            index=False
        )

        print(f"\nSelected Features (threshold = {threshold})")

        for feature in self.selected_features:
            print(f"- {feature}")

        selected_df = self.transformed_df[
            self.selected_features + [self.target_col]
        ]

        return selected_df

    # 4. Run Complete Feature Selection

    def run_feature_selection(self, threshold=0.05):

        self.corr_heatmap()

        self.compute_correlation()

        selected_df = self.select_features(threshold)

        print("\nFeature Selection completed successfully.")

        return selected_df, self.selected_features