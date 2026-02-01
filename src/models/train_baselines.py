import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from src.data.make_dataset import load_dataset, prepare_dataframe, TARGET
from src.data.preprocess import build_preprocessor_linear

DATA_PATH = "data/CommViolPredUnnormalizedData.txt"


def split(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)
    return x_train, x_val, x_test, y_train, y_val, y_test


def eval_reg(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    df = load_dataset(DATA_PATH)
    df = prepare_dataframe(df)

    x= df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

    pre = build_preprocessor_linear(
        df, imputer="median", scaler="standard", missing_threshold=0.80
    )

    enet = ElasticNet(
    max_iter=200000, tol=1e-4, selection="random", random_state=42
    )

    pipe = Pipeline([("prep", pre), ("model", enet)])

    param_grid = {
        "model__alpha": np.logspace(-4, 1, 12),      # 1e-4 ... 10
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9] # mix ridge<->lasso
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1
    )

    gs.fit(x_train, y_train)

    print("\n=== ENET GRIDSEARCH (CV on TRAIN) ===")
    print("Best params:", gs.best_params_)
    print("Best CV RMSE:", -gs.best_score_)

    best_pipe = gs.best_estimator_

    pred_val = best_pipe.predict(x_val)
    rmse, mae, r2 = eval_reg(y_val, pred_val)
    print("\n=== ENET BEST (VALIDATION) ===")
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    pred_test = best_pipe.predict(x_test)
    rmse, mae, r2 = eval_reg(y_test, pred_test)
    print("\n=== ENET BEST (TEST) ===")
    print(f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.01, max_iter=100000, tol=1e-4, selection="random", random_state=42),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000, tol=1e-4, selection="random", random_state=42),
    }

    results = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(x_train, y_train)

        pred_val = pipe.predict(x_val)
        rmse, mae, r2 = eval_reg(y_val, pred_val)
        results[name] = (rmse, mae, r2)

    print("=== VALIDATION RESULTS ===")
    for name, (rmse, mae, r2) in results.items():
        print(f"{name:20s} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    best_name = min(results, key=lambda k: results[k][0])
    print(f"\nBest on VAL: {best_name}")

    best_model = models[best_name]
    best_pipe = Pipeline([("prep", pre), ("model", best_model)])
    best_pipe.fit(x_train, y_train)
    pred_test = best_pipe.predict(x_test)
    rmse, mae, r2 = eval_reg(y_test, pred_test)
    print("\n=== TEST (FINAL CHECK) ===")
    print(f"{best_name:20s} RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")


if __name__ == "__main__":
    main()
