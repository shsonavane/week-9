import pandas as pd
import numpy as np


class GroupEstimate:
    """
    A simple estimator that groups categorical features
    and predicts numeric outcomes using mean or median.
    """

    def __init__(self, estimate="mean"):
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates = None

    def fit(self, X, y):
        """
        Fit the model by calculating group-wise estimates.

        X: pandas DataFrame of categorical variables.
        y: pandas Series or 1D array of numeric targets.
        """
        # combine input data
        df = pd.concat([X.reset_index(drop=True),
                        pd.Series(y, name="target")], axis=1)

        # choose aggregation function
        agg_func = np.mean if self.estimate == "mean" else np.median

        # compute group-level estimates
        self.group_estimates = (
            df.groupby(list(X.columns))["target"]
            .agg(agg_func)
            .reset_index()
        )

    def predict(self, X_):
        """
        Predict estimated values for new observations.
        Returns an array of predicted y values or NaN if group not seen.
        """
        if self.group_estimates is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # ensure input is DataFrame
        X_ = pd.DataFrame(X_, columns=self.group_estimates.columns[:-1])

        # merge on group keys
        merged = X_.merge(self.group_estimates,
                          on=list(X_.columns),
                          how="left")

        missing = merged["target"].isna().sum()
        if missing > 0:
            print(f"{missing} unseen group(s) encountered.")

        return merged["target"].values
