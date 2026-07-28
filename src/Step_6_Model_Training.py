import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os


class ModelTrainer:
    def __init__(self, X_train, y_train, output_dir="./outputs/Models"):
        """
        X_train, y_train: training data
        output_dir: folder to save trained models
        """
        self.X_train = X_train
        self.y_train = y_train
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.rf_model = None
        self.gbr_model = None

    # Train Random Forest
    def train_random_forest(self, rf_params=None, model_file_name="rf_cost_predictor.pkl"):
        if rf_params is None:
            rf_params = {"n_estimators": 200, "max_depth": None, "random_state": 42}

        self.rf_model = RandomForestRegressor(**rf_params, n_jobs=-1)
        self.rf_model.fit(self.X_train, self.y_train)

        rf_path = os.path.join(self.output_dir, model_file_name)
        joblib.dump(self.rf_model, rf_path)
        print(f"Random Forest model saved as {rf_path}")
        return self.rf_model

    # Train Gradient Boosting
    def train_gradient_boosting(self, gbr_params=None, model_file_name="gb_cost_predictor.pkl"):
        if gbr_params is None:
            gbr_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}

        self.gbr_model = GradientBoostingRegressor(**gbr_params)
        self.gbr_model.fit(self.X_train, self.y_train)

        gbr_path = os.path.join(self.output_dir, model_file_name)
        joblib.dump(self.gbr_model, gbr_path)
        print(f"Gradient Boosting Regression model saved as {gbr_path}")
        return self.gbr_model

    # Train specified model only
    def train_model(self, model_name, params=None, model_file_name=None):
        """
        model_name: "rf" for Random Forest, "gbr" for Gradient Boosting
        params: dictionary of hyperparameters for the model
        model_file_name: optional file name to save the model
        """
        if model_name.lower() == "rf":
            return self.train_random_forest(rf_params=params,
                                            model_file_name=model_file_name or "rf_cost_predictor.pkl")
        elif model_name.lower() == "gbr":
            return self.train_gradient_boosting(gbr_params=params,
                                                model_file_name=model_file_name or "gb_cost_predictor.pkl")
        else:
            raise ValueError("Invalid model_name. Use 'rf' or 'gbr'.")