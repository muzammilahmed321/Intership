import os
import pandas as pd
import matplotlib.pyplot as plt

class ModelComparator:
    def __init__(self, output_dir="./outputs/ModelComparison"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def compare(self, metrics_list):
        """
        metrics_list: list of dicts, each like
        {"Model": "rf", "RMSE": .., "MAE": .., "R2_Score": ..}
        """
        comparison_df = pd.DataFrame(metrics_list)
        comparison_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
        print("\nModel Comparison:\n", comparison_df)

        for metric in ["RMSE", "MAE", "R2_Score"]:
            plt.figure(figsize=(6, 4))
            plt.bar(comparison_df["Model"], comparison_df[metric])
            plt.title(f"{metric} Comparison")
            plt.ylabel(metric)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{metric}_comparison.png"))
            plt.show()
            plt.close()

        return comparison_df