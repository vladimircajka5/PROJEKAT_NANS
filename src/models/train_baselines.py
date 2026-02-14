import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.data.make_dataset import load_dataset, prepare_dataframe, TARGET
from src.data.preprocess import build_preprocessor_linear

from src.models.result import FitResult

DATA_PATH = "data/CommViolPredUnnormalizedData.txt"
RANDOM_STATE = 42

CV_FOLDS = 5

def split(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def eval_reg(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, r2


def cv_rmse_kfold(pipe, x_train, y_train, folds=CV_FOLDS, n_jobs=1):

    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe,
        x_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs
    )
    return float(-scores.mean())


def fit_and_score(
    name, model, pre,
    x_train, y_train, x_val, y_val, x_test, y_test,
    imputer="median", scaler="standard",
    cv_folds=CV_FOLDS
):
    pipe = Pipeline([("prep", pre), ("model", model)])

    cv_rmse = cv_rmse_kfold(pipe, x_train, y_train, folds=cv_folds, n_jobs=1)

    pipe.fit(x_train, y_train)

    pred_val = pipe.predict(x_val)
    val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

    pred_test = pipe.predict(x_test)
    test_rmse, test_mae, test_r2 = eval_reg(y_test, pred_test)

    return FitResult(
        name=name, imputer=imputer, scaler=scaler,
        is_tuned=False, best_params=None, cv_rmse=cv_rmse,
        val_rmse=val_rmse, val_mae=val_mae, val_r2=val_r2,
        test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2
    )


def tune_and_score(
    name, base_model, pre, param_grid,
    x_train, y_train, x_val, y_val, x_test, y_test,
    imputer="median", scaler="standard", cv=5
):
    pipe = Pipeline([("prep", pre), ("model", base_model)])

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1
    )
    gs.fit(x_train, y_train)

    best_pipe = gs.best_estimator_
    cv_rmse = float(-gs.best_score_)  # neg -> pos

    pred_val = best_pipe.predict(x_val)
    val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

    pred_test = best_pipe.predict(x_test)
    test_rmse, test_mae, test_r2 = eval_reg(y_test, pred_test)

    return FitResult(
        name=name, imputer=imputer, scaler=scaler,
        is_tuned=True, best_params=gs.best_params_, cv_rmse=cv_rmse,
        val_rmse=val_rmse, val_mae=val_mae, val_r2=val_r2,
        test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2
    ), best_pipe


def main():
    df = load_dataset(DATA_PATH)
    df = prepare_dataframe(df)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

    all_results = []

    variants = [(imp, sc) for imp in ["median", "knn", "mice"] for sc in ["standard", "robust"]]

    baseline_models = [
        ("OLS", LinearRegression()),
        ("Ridge(alpha=1)", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ("Lasso(alpha=0.01)", Lasso(alpha=0.01, max_iter=300000, tol=1e-4,
                                    selection="random", random_state=RANDOM_STATE)),
        ("ElasticNet(a=0.01,l1=0.5)", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=300000, tol=1e-4,
                                                selection="random", random_state=RANDOM_STATE)),
    ]

    for imputer, scaler in variants:
        pre = build_preprocessor_linear(df, imputer=imputer, scaler=scaler, missing_threshold=0.80)
        for name, model in baseline_models:
            all_results.append(
                fit_and_score(
                    name, model, pre,
                    x_train, y_train, x_val, y_val, x_test, y_test,
                    imputer, scaler,
                    cv_folds=CV_FOLDS, n_jobs=1
                )
            )

    df_res = pd.DataFrame([r.__dict__ for r in all_results]).sort_values(["val_rmse", "test_rmse"]).reset_index(drop=True)

    print("\n=== TOP 15 BY VAL RMSE (baseline sweep) ===")
    show_cols = ["name", "imputer", "scaler", "cv_rmse", "val_rmse", "val_mae", "val_r2", "test_rmse", "test_r2"]
    print(df_res[show_cols].head(15).to_string(index=False))

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results_linear_baselines.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    best_row = df_res.iloc[0]
    best_imputer = best_row["imputer"]
    best_scaler = best_row["scaler"]
    print("\nBest preprocessing (by VAL RMSE):", {"imputer": best_imputer, "scaler": best_scaler})

    pre_best = build_preprocessor_linear(df, imputer=best_imputer, scaler=best_scaler, missing_threshold=0.80)

    enet_base = ElasticNet(
        max_iter=500000, tol=1e-4, selection="random", random_state=RANDOM_STATE
    )

    param_grid = {
        "model__alpha": np.logspace(-3, 2, 10),
        "model__l1_ratio": [0.2, 0.5, 0.8, 0.9],
    }

    tuned_result, tuned_pipe = tune_and_score(
        "ElasticNet (tuned)", enet_base, pre_best, param_grid,
        x_train, y_train, x_val, y_val, x_test, y_test,
        imputer=best_imputer, scaler=best_scaler, cv=5
    )

    print("\n=== ENET GRIDSEARCH (CV on TRAIN) ===")
    print("Best params:", tuned_result.best_params)
    print("Best CV RMSE:", tuned_result.cv_rmse)

    print("\n=== ENET TUNED (VALIDATION) ===")
    print(f"RMSE={tuned_result.val_rmse:.4f}  MAE={tuned_result.val_mae:.4f}  R2={tuned_result.val_r2:.4f}")

    print("\n=== ENET TUNED (TEST - final) ===")
    print(f"RMSE={tuned_result.test_rmse:.4f}  MAE={tuned_result.test_mae:.4f}  R2={tuned_result.test_r2:.4f}")

    y_hat_val = tuned_pipe.predict(x_val)

    plt.figure()
    plt.scatter(y_val, y_hat_val)
    mn = min(float(y_val.min()), float(y_hat_val.min()))
    mx = max(float(y_val.max()), float(y_hat_val.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.title("Validation: y_true vs y_pred (ElasticNet tuned)")
    plt.show()


if __name__ == "__main__":
    main()
