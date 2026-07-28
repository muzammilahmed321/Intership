import numpy as np
from sklearn.model_selection import cross_val_score, KFold


class CrossValidation:
    def __init__(self, model, X, y, model_name, cv_folds=5):
        self.model = model
        self.X = X
        self.y = y
        self.model_name = model_name
        self.cv_folds = cv_folds

    def run_cv(self):
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.model, self.X, self.y, cv=kfold,
            scoring="neg_root_mean_squared_error"
        )
        rmse_scores = -scores

        print(f"\n{self.model_name} Cross-Validation RMSE per fold:", rmse_scores)
        print(f"Mean RMSE: {rmse_scores.mean():.2f}")
        print(f"Std RMSE: {rmse_scores.std():.2f}")

        return rmse_scores.mean(), rmse_scores.std()
