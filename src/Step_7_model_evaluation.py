import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ModelEvaluator:
    def __init__(
        self,
        model,
        X_test,
        y_test,
        model_name,
        output_dir="./outputs/ModelTraining",
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\n{'=' * 40}")
        print(f"Model: {self.model_name}")
        print(f"{'=' * 40}")
        print(f"RMSE     : {rmse:.2f}")
        print(f"MAE      : {mae:.2f}")
        print(f"R² Score : {r2:.4f}")
        print(f"{'=' * 40}\n")

        metrics = {
            "Model": self.model_name,
            "RMSE": rmse,
            "MAE": mae,
            "R2_Score": r2,
        }

        # Actual vs Predicted Scatter Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.y_test,
            y_pred,
            alpha=0.7,
            edgecolors="black",
        )

        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())

        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        plt.xlabel("Actual Charges")
        plt.ylabel("Predicted Charges")
        plt.title(f"{self.model_name} - Actual vs Predicted")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            f"{self.model_name}_actual_vs_predicted.png",
        )

        # Save plot
        plt.savefig(plot_path, dpi=300)
        plt.show()
        plt.close()

        return metrics, y_pred