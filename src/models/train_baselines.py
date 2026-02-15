import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
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

def ols_numerical_diagnostics(pre, x_train, y_train):

    X = pre.fit_transform(x_train, y_train)

    try:
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    # singularnost
    s = np.linalg.svd(X, compute_uv=False)
    s_min = float(s.min())
    s_max = float(s.max())
    cond = float(s_max / s_min) if s_min > 0 else float("inf")

    # kondicioni broj
    rank = int(np.linalg.matrix_rank(X))

    return {"rank": rank, "s_min": s_min, "cond": cond, "n_features": int(X.shape[1])}



def cv_rmse_kfold(pipe, x_train, y_train, folds=CV_FOLDS, n_jobs=-1):

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

    cv_rmse = cv_rmse_kfold(pipe, x_train, y_train, folds=cv_folds, n_jobs=-1)

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
        n_jobs=-1
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

    pre_ols_none = build_preprocessor_linear(df, imputer="median", scaler="none", missing_threshold=0.80)
    pre_ols_std  = build_preprocessor_linear(df, imputer="median", scaler="standard", missing_threshold=0.80)

    diag_none = ols_numerical_diagnostics(pre_ols_none, x_train, y_train)
    diag_std  = ols_numerical_diagnostics(pre_ols_std, x_train, y_train)

    csv_dir = Path("results/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)

    df_diag = pd.DataFrame([
        {"variant": "OLS_median_none", **diag_none},
        {"variant": "OLS_median_standard", **diag_std},
    ])
    diag_path = csv_dir / "ols_diagnostics.csv"
    df_diag.to_csv(diag_path, index=False)
    print(f"Saved: {diag_path}")

    print("\n=== OLS NUMERICAL DIAGNOSTICS (TRAIN) ===")
    print("OLS (median, no scaling):   ", diag_none)
    print("OLS (median, z-score):      ", diag_std)


    all_results = []

    variants = [(imp, sc) for imp in ["median", "knn", "mice"] for sc in ["none", "standard", "robust"]]

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
            
            if scaler == "none" and name in ["Lasso(alpha=0.01)", "ElasticNet(a=0.01,l1=0.5)"]:
                continue

            all_results.append(
                fit_and_score(
                    name, model, pre,
                    x_train, y_train, x_val, y_val, x_test, y_test,
                    imputer, scaler,
                    cv_folds=CV_FOLDS
                )
            )

    df_res = pd.DataFrame([r.__dict__ for r in all_results]).sort_values(["val_rmse", "test_rmse"]).reset_index(drop=True)

    print("\n=== TOP 15 BY VAL RMSE (baseline sweep) ===")
    show_cols = ["name", "imputer", "scaler", "cv_rmse", "val_rmse", "val_mae", "val_r2", "test_rmse", "test_r2"]
    print(df_res[show_cols].head(15).to_string(index=False))

    csv_dir = Path("results/csv")
    plot_dir = Path("results/plots")
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = csv_dir / "results_linear_baselines.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    best_row = df_res.iloc[0]
    best_imputer = best_row["imputer"]
    best_scaler = best_row["scaler"]
    print("\nBest preprocessing (by VAL RMSE):", {"imputer": best_imputer, "scaler": best_scaler})

    pre_best = build_preprocessor_linear(df, imputer=best_imputer, scaler=best_scaler, missing_threshold=0.80)

    # Ridge tuning (L2)
    ridge_base = Ridge(random_state=RANDOM_STATE)
    ridge_grid = {
        "model__alpha": np.logspace(-3, 3, 15),
    }
    tuned_ridge, ridge_pipe = tune_and_score(
        "Ridge (tuned)", ridge_base, pre_best, ridge_grid,
        x_train, y_train, x_val, y_val, x_test, y_test,
        imputer=best_imputer, scaler=best_scaler, cv=CV_FOLDS
    )

    print("\n=== RIDGE GRIDSEARCH (CV on TRAIN) ===")
    print("Best params:", tuned_ridge.best_params)
    print(f"CV RMSE={tuned_ridge.cv_rmse:.4f}")
    print(f"VAL  : RMSE={tuned_ridge.val_rmse:.4f}  MAE={tuned_ridge.val_mae:.4f}  R2={tuned_ridge.val_r2:.4f}")
    print(f"TEST : RMSE={tuned_ridge.test_rmse:.4f}  MAE={tuned_ridge.test_mae:.4f}  R2={tuned_ridge.test_r2:.4f}")

    # Lasso tuning (L1)
    lasso_base = Lasso(
        max_iter=500000, tol=1e-4, selection="random", random_state=RANDOM_STATE
    )
    lasso_grid = {
        "model__alpha": np.logspace(-3, 2, 15),
    }
    tuned_lasso, lasso_pipe = tune_and_score(
        "Lasso (tuned)", lasso_base, pre_best, lasso_grid,
        x_train, y_train, x_val, y_val, x_test, y_test,
        imputer=best_imputer, scaler=best_scaler, cv=CV_FOLDS
    )

    print("\n=== LASSO GRIDSEARCH (CV on TRAIN) ===")
    print("Best params:", tuned_lasso.best_params)
    print(f"CV RMSE={tuned_lasso.cv_rmse:.4f}")
    print(f"VAL  : RMSE={tuned_lasso.val_rmse:.4f}  MAE={tuned_lasso.val_mae:.4f}  R2={tuned_lasso.val_r2:.4f}")
    print(f"TEST : RMSE={tuned_lasso.test_rmse:.4f}  MAE={tuned_lasso.test_mae:.4f}  R2={tuned_lasso.test_r2:.4f}")

    # Elastic Net tuning (L1+L2)
    enet_base = ElasticNet(
        max_iter=500000, tol=1e-4, selection="random", random_state=RANDOM_STATE
    )

    enet_grid = {
        "model__alpha": np.logspace(-3, 2, 10),
        "model__l1_ratio": [0.1, 0.2, 0.5, 0.8, 0.9, 0.95],
    }

    tuned_enet, enet_pipe = tune_and_score(
        "ElasticNet (tuned)", enet_base, pre_best, enet_grid,
        x_train, y_train, x_val, y_val, x_test, y_test,
        imputer=best_imputer, scaler=best_scaler, cv=CV_FOLDS
    )

    print("\n=== ELASTIC NET GRIDSEARCH (CV on TRAIN) ===")
    print("Best params:", tuned_enet.best_params)
    print(f"CV RMSE={tuned_enet.cv_rmse:.4f}")
    print(f"VAL  : RMSE={tuned_enet.val_rmse:.4f}  MAE={tuned_enet.val_mae:.4f}  R2={tuned_enet.val_r2:.4f}")
    print(f"TEST : RMSE={tuned_enet.test_rmse:.4f}  MAE={tuned_enet.test_mae:.4f}  R2={tuned_enet.test_r2:.4f}")

    # tabela svih tuned modela
    tuned_results = [tuned_ridge, tuned_lasso, tuned_enet]
    df_tuned = pd.DataFrame([r.__dict__ for r in tuned_results]).sort_values("cv_rmse").reset_index(drop=True)

    print("\n=== REGULARIZED MODELS COMPARISON (sorted by CV RMSE) ===")
    print(df_tuned[show_cols].to_string(index=False))

    # dodaj tuned rezultate u ukupnu tabelu
    all_results.extend(tuned_results)
    df_all = pd.DataFrame([r.__dict__ for r in all_results]).sort_values(["val_rmse", "test_rmse"]).reset_index(drop=True)

    out_path_all = csv_dir / "results_linear_baselines.csv"
    df_all.to_csv(out_path_all, index=False)
    print(f"\nSaved: {out_path_all}")

    # scatter za best tuned regularizovani model
    best_tuned = df_tuned.iloc[0]
    best_tuned_name = best_tuned["name"]
    if best_tuned_name == "Ridge (tuned)":
        best_tuned_pipe = ridge_pipe
    elif best_tuned_name == "Lasso (tuned)":
        best_tuned_pipe = lasso_pipe
    else:
        best_tuned_pipe = enet_pipe

    y_hat_val = best_tuned_pipe.predict(x_val)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_val, y_hat_val, alpha=0.5, s=18, edgecolors="k", linewidths=0.3)
    mn = min(float(y_val.min()), float(y_hat_val.min()))
    mx = max(float(y_val.max()), float(y_hat_val.max()))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    plt.xlabel("y_true (ViolentCrimesPerPop)")
    plt.ylabel("y_pred")
    plt.title(f"Validation: y_true vs y_pred ({best_tuned_name})")
    plt.tight_layout()
    plt.savefig(plot_dir / "baseline_pred_vs_true_val.png", dpi=160)
    plt.close()
    print(f"Saved: {plot_dir / 'baseline_pred_vs_true_val.png'}")


if __name__ == "__main__":
    main()
