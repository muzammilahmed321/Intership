from sklearn.model_selection import GridSearchCV

class HyperparameterTuner:
    def __init__(self, model, param_grid, X_train, y_train, model_name, cv_folds=5):
        self.model = model
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.cv_folds = cv_folds

    def run_grid_search(self):
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(self.X_train, self.y_train)

        print(f"\n{self.model_name} Best Parameters:", grid.best_params_)
        print(f"{self.model_name} Best CV RMSE: {-grid.best_score_:.2f}")

        return grid.best_estimator_, grid.best_params_