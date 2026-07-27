import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import norm


class EDA:

    def __init__(self, insurance_df_filled, output_dir="./EDA"):
        """
        insurance_df_filled: cleaned insurance dataset (output of DataPreparation)
        output_dir: folder to save EDA results and plots
        """
        self.insurance_df_filled = insurance_df_filled
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # 1. Basic Info & Statistical Description
    def describe_data(self):
        self.insurance_df_filled.info()
        desc = self.insurance_df_filled.describe()
        desc.to_csv(os.path.join(self.output_dir, "insurance_df_description.csv"))
        return desc

    # 2. Count Plots for Categorical Features
    def plot_categorical_counts(self):
        cat_columns = {
            "sex": "Count of Customers by Sex",
            "smoker": "Count of Customers by Smoker Status",
            "region": "Count of Customers by Region",
            "children": "Count of Customers by Number of Children"
        }

        for col, title in cat_columns.items():
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, data=self.insurance_df_filled)
            plt.title(title, size=16)
            plt.ylabel("Count", size=12)
            plt.xlabel(col, size=12)
            sns.despine(top=True, right=True, left=False, bottom=False)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"Countof_{col}_graph.png"))
            plt.show()
            plt.close()

    # 3. Charges Distribution
    def plot_charges_distributions(self):
        df = self.insurance_df_filled

        # Histogram with bell curve overlay
        plt.figure(figsize=(8, 5))
        plt.hist(df["charges"], bins=50, density=True, color='skyblue', edgecolor='black')
        mu, std = df["charges"].mean(), df["charges"].std()
        x = np.linspace(df["charges"].min(), df["charges"].max(), 100)
        plt.plot(x, norm.pdf(x, mu, std), color='red', lw=2, label='Normal Curve')
        plt.xlabel("Charges (USD)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Insurance Charges")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "charges_hist.png"))
        plt.show()
        plt.close()

        # Charges by smoker status
        plt.figure(figsize=(8, 5))
        sns.barplot(x='smoker', y='charges', data=df)
        plt.title("Average Charges by Smoker Status")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ChargesBySmoker.png"))
        plt.show()
        plt.close()

        # Charges by region
        plt.figure(figsize=(8, 5))
        sns.barplot(x='region', y='charges', data=df)
        plt.title("Average Charges by Region")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ChargesByRegion.png"))
        plt.show()
        plt.close()

        # Charges vs age, colored by smoker status
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='age', y='charges', hue='smoker', data=df)
        plt.title("Charges vs Age by Smoker Status")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "ChargesVsAgeBySmoker.png"))
        plt.show()
        plt.close()

        # Average charges per smoker status and region
        avg_charges = df.groupby(['smoker', 'region'])['charges'].mean().reset_index()
        avg_charges.to_csv(os.path.join(self.output_dir, "AverageChargesBySmokerAndRegion.csv"), index=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='region', y='charges', hue='smoker', data=avg_charges)
        plt.title("Average Charges by Smoker Status and Region")
        plt.ylabel("Average Charges (USD)")
        plt.xlabel("Region")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "AverageChargesBySmokerAndRegion.png"))
        plt.show()
        plt.close()

    # 4. Numeric Feature Analysis
    def analyze_numeric_features(self):
        df = self.insurance_df_filled
        numeric_cols = df.select_dtypes(include=np.number).columns

        # Histograms with bell curve for each numeric feature
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        plt.figure(figsize=(18, 5 * n_rows))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            data = df[col].dropna()
            sns.histplot(data, bins=30, stat='density', color='skyblue', edgecolor='black')
            mu, std = data.mean(), data.std()
            x = np.linspace(data.min(), data.max(), 100)
            plt.plot(x, norm.pdf(x, mu, std), color='red', lw=2, label='Normal Curve')
            plt.title(col)
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.legend()

        plt.suptitle("Distribution of Numeric Features with Bell Curve", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, "Numeric_hist.png"))
        plt.show()
        plt.close()

        # Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[numeric_cols])
        plt.title("Boxplot of Numeric Features")
        plt.savefig(os.path.join(self.output_dir, "Numeric_Boxplot.png"))
        plt.show()
        plt.close()

        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation of Numeric Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "Numeric_Correlation.png"))
        plt.show()
        plt.close()

        # Pairplot
        sns.pairplot(df[numeric_cols])
        plt.savefig(os.path.join(self.output_dir, "Numeric_Pairplot.png"))
        plt.show()
        plt.close()

    # 5. Run Full EDA
    def run_eda(self):
        self.describe_data()
        self.plot_categorical_counts()
        self.plot_charges_distributions()
        self.analyze_numeric_features()
        print("\n EDA completed successfully.")