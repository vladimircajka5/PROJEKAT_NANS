from src.data.catboost_imputer import CatBoostDfImputer
from src.data.make_dataset import TARGET

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.80, target=None):
        self.threshold = threshold
        self.target = target
        self.to_drop_ = None

    def fit(self, x, y=None):
        missing_ratio = x.isna().mean()
        to_drop = []
        for col, ratio in missing_ratio.items():
            if self.target is not None and col == self.target:
                continue
            if ratio > self.threshold:
                to_drop.append(col)
        self.to_drop_ = to_drop
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors="ignore")

def build_preprocessor_linear(df, imputer="median", scaler="standard", missing_threshold=0.80):
    if imputer == "median":
        num_imputer = SimpleImputer(strategy="median")
    elif imputer == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif imputer == "mice":
        num_imputer = IterativeImputer(random_state=42)
    else:
        raise ValueError("imputer must be: median | knn | mice")

    if scaler == "standard":
        scaler_step = StandardScaler()
    elif scaler == "robust":
        scaler_step = RobustScaler()
    else:
        raise ValueError("scaler must be: standard | robust")

    num = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler_step),
    ])

    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_selector = lambda x: x.select_dtypes(exclude=["object", "category"]).columns
    cat_selector = lambda x: x.select_dtypes(include=["object", "category"]).columns


    pre = Pipeline([
        ("drop_target", FunctionTransformer(lambda x: x.drop(columns=[TARGET], errors="ignore"))),

        ("drop_high_missing", DropHighMissing(threshold=missing_threshold, target=TARGET)),

        ("ct", ColumnTransformer(
            [("num", num, num_selector), ("cat", cat, cat_selector)],
            remainder="drop",
        )),
    ])
    return pre



def build_preprocessor_catboost(df, imputer="median", missing_threshold=0.80):
    if imputer not in ["median", "knn", "mice"]:
        raise ValueError("imputer must be: median | knn | mice")

    num_strategy = imputer

    pre = Pipeline([
        ("drop_target", FunctionTransformer(lambda x: x.drop(columns=[TARGET], errors="ignore"))),
        ("drop_high_missing", DropHighMissing(threshold=missing_threshold, target=TARGET)),
        ("df_imputer", CatBoostDfImputer(num_strategy=num_strategy)),
    ])
    return pre

