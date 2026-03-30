import numpy as np
import pandas as pd


class GroupEstimate(object):
    """A grouped estimator for categorical features and continuous targets."""

    def __init__(self, estimate):
        """
        Initialize the GroupEstimate model.

        Parameters
        ----------
        estimate : str
            Aggregation method to use ('mean' or 'median').
        """
        if estimate not in {"mean", "median"}:
            raise ValueError("estimate must be 'mean' or 'median'")

        self.estimate = estimate
        self.group_map = None
        self.columns = None
        self.default_map = None
        self.default_category = None

    def fit(self, X, y, default_category=None):
        """
        Fit the model using categorical features and a continuous target.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing categorical features.
        y : array-like
            Continuous target values corresponding to rows in X.
        default_category : str, optional
            Column name to use for fallback grouping when combinations
            are missing.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.columns = list(X.columns)
        self.default_category = default_category

        df = X.copy()
        df["_target"] = y

        agg_func = "mean" if self.estimate == "mean" else "median"

        grouped = df.groupby(self.columns, observed=True)["_target"].agg(agg_func)
        self.group_map = grouped.to_dict()

        if default_category is not None:
            if default_category not in self.columns:
                raise ValueError("default_category must be a column in X")

            default_grouped = df.groupby(
                default_category, observed=True
            )["_target"].agg(agg_func)

            self.default_map = default_grouped.to_dict()

    def predict(self, X):
        """
        Predict target values for new observations.

        Parameters
        ----------
        X : array-like or pandas.DataFrame
            New categorical observations.

        Returns
        -------
        list
            Predicted values based on learned group estimates.
        """
        if self.group_map is None:
            raise ValueError("Model has not been fitted yet")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns)

        predictions = []
        missing_count = 0

        for _, row in X.iterrows():
            key = tuple(row[col] for col in self.columns)

            if key in self.group_map:
                predictions.append(self.group_map[key])
            else:
                if self.default_map is not None:
                    fallback_value = row[self.default_category]

                    if fallback_value in self.default_map:
                        predictions.append(
                            self.default_map[fallback_value]
                        )
                        continue

                predictions.append(np.nan)
                missing_count += 1

        if missing_count > 0:
            print(f"{missing_count} missing group(s) encountered.")

        return predictions