import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, selected_df, target_col="charges", random_state=42):
        """
        selected_df : DataFrame after feature selection
        target_col  : Target column name
        random_state: Random seed
        """

        self.selected_df = selected_df.copy()
        self.target_col = target_col
        self.random_state = random_state

        # Outputs
        self.X = None
        self.Y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    # Prepare features and target
    def prepare_features_target(self):
        self.X = self.selected_df.drop(columns=[self.target_col])
        self.Y = self.selected_df[self.target_col]

        return self.X, self.Y

    # Split dataset
    def split_train_test(self, test_size=0.2):

        if self.X is None or self.Y is None:
            self.prepare_features_target()

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.Y,
            test_size=test_size,
            random_state=self.random_state,
        )

        print(f"X_train shape: {self.X_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"y_test shape: {self.y_test.shape}")

        return (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        )

    # Run complete splitting pipeline
    def run_split(self):
        self.prepare_features_target()
        return self.split_train_test()