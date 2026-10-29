import pandas as pd


class GroupEstimate(object):
    def __init__(self, estimate):
        allowed = {"mean", "median"}
        if estimate not in allowed:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        y = pd.Series(y, index=X.index)
        if y.isna().any():
            raise ValueError("y cannot contain missing values")

        combined = X.copy()
        combined["y"] = y.to_numpy()

        self.columns_ = list(X.columns)
        self.group_estimates_ = (
            combined.groupby(self.columns_, dropna=False)["y"]
            .agg(self.estimate)
            .rename("estimate")
        )

        return self

    def predict(self, X_):
        return None