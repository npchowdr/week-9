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
        if not hasattr(self, "group_estimates_"):
            raise ValueError("Model must be fit before calling predict")

        if isinstance(X_, pd.DataFrame):
            X_pred = X_.copy()
        else:
            X_pred = pd.DataFrame(X_, columns=self.columns_)

        if list(X_pred.columns) != self.columns_:
            X_pred = X_pred.loc[:, self.columns_]

        if len(self.columns_) == 1:
            keys = X_pred[self.columns_[0]]
            predictions = keys.map(self.group_estimates_)
        else:
            keys = X_pred.apply(tuple, axis=1)
            predictions = keys.map(self.group_estimates_)

        missing_count = int(predictions.isna().sum())
        if missing_count > 0:
            print(f"Number of missing groups: {missing_count}")

        return predictions.to_numpy(dtype=float)