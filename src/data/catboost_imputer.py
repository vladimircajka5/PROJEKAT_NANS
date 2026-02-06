from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class CatBoostDfImputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_strategy="median"):
        self.num_strategy = num_strategy
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.num_cols_ = None
        self.cat_cols_ = None

    def fit(self, x, y=None):
        self.num_cols_ = x.select_dtypes(exclude=["object", "category"]).columns.tolist()
        self.cat_cols_ = x.select_dtypes(include=["object", "category"]).columns.tolist()

        self.num_imputer_ = SimpleImputer(strategy=self.num_strategy)
        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")

        if self.num_cols_:
            self.num_imputer_.fit(x[self.num_cols_])
        if self.cat_cols_:
            self.cat_imputer_.fit(x[self.cat_cols_])

        return self

    def transform(self, x):
        x = x.copy()

        if self.num_cols_:
            x[self.num_cols_] = self.num_imputer_.transform(x[self.num_cols_])
        if self.cat_cols_:
            x[self.cat_cols_] = self.cat_imputer_.transform(x[self.cat_cols_])
            # CatBoost voli string kategorije
            for c in self.cat_cols_:
                x[c] = x[c].astype("string").fillna("NA")

        return x
