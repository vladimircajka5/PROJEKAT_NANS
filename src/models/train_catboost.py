import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor

from src.data.make_dataset import load_dataset, prepare_dataframe, TARGET
from src.data.preprocess import build_preprocessor_catboost
from src.models.result import FitResult

DATA_PATH = "data/CommViolPredUnnormalizedData.txt"
RANDOM_STATE = 42


def split(x, y, random_state=RANDOM_STATE):
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.20, random_state=random_state
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def eval_reg(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def main():
    df = load_dataset(DATA_PATH)
    df = prepare_dataframe(df)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

    imputer = "median"
    pre = build_preprocessor_catboost(df, imputer=imputer, missing_threshold=0.80)

    # fit preprocessor samo na train
    pre.fit(x_train, y_train)

    xtr = pre.transform(x_train)
    xva = pre.transform(x_val)
    xte = pre.transform(x_test)

    cat_cols = xtr.select_dtypes(include=["object","category","string"]).columns.tolist()
    cat_features = [xtr.columns.get_loc(c) for c in cat_cols]


    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.8,
        early_stopping_rounds=100,
        verbose=200
    )

    model.fit(
        xtr, y_train,
        eval_set=(xva, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    pred_val = model.predict(xva)
    val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

    pred_test = model.predict(xte)
    test_rmse, test_mae, test_r2 = eval_reg(y_test, pred_test)

    res = FitResult(
        name="CatBoost",
        imputer=imputer,
        scaler="none",
        is_tuned=False,
        best_params={
            "best_iteration": model.get_best_iteration(),
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
            "colsample_bylevel": 0.8,
        },
        cv_rmse=None,
        val_rmse=val_rmse, val_mae=val_mae, val_r2=val_r2,
        test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2
    )

    print("\n=== CATBOOST RESULTS ===")
    print(f"VAL  RMSE={res.val_rmse:.4f}  MAE={res.val_mae:.4f}  R2={res.val_r2:.4f}")
    print(f"TEST RMSE={res.test_rmse:.4f}  MAE={res.test_mae:.4f}  R2={res.test_r2:.4f}")
    print("Best iteration:", model.get_best_iteration())

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_catboost.csv"
    pd.DataFrame([res.__dict__]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
