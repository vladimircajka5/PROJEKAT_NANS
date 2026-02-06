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


def fmt_float(x, nd=4):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def print_table(rows, cols, title=None):
    if title:
        print("\n" + title)
    if not rows:
        print("(no rows)")
        return

    str_rows = [[str(r.get(c, "")) for c in cols] for r in rows]

    widths = []
    for j, c in enumerate(cols):
        widths.append(max(len(c), max(len(row[j]) for row in str_rows)))

    header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)

    print(header)
    print(sep)
    for row in str_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(cols))))


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

    cat_cols = xtr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_features = [xtr.columns.get_loc(c) for c in cat_cols]

    #tuning (biramo najbolje hiperparametre na osnovu VAL RMSE)
    grid = [
        {"depth": 6,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.03, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.10, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 1.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 5.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    ]

    best_row = None
    best_model = None
    tuning_rows = []

    print("\n=== CATBOOST TUNING (select by VAL RMSE) ===")
    for i, g in enumerate(grid, start=1):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            iterations=8000,
            learning_rate=g["learning_rate"],
            depth=g["depth"],
            l2_leaf_reg=g["l2_leaf_reg"],
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=200,
            verbose=200,
        )

        model.fit(
            xtr, y_train,
            eval_set=(xva, y_val),
            cat_features=cat_features,
            use_best_model=True
        )

        pred_val = model.predict(xva)
        val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

        row = {
            **g,
            "best_iteration": int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "val_r2": float(val_r2),
        }

        print(
            f"[{i}/{len(grid)}] depth={row['depth']} "
            f"lr={row['learning_rate']} l2={row['l2_leaf_reg']} "
            f"-> VAL_RMSE={row['val_rmse']:.4f}"
        )

        tuning_rows.append({
            "run": i,
            "depth": row["depth"],
            "lr": row["learning_rate"],
            "l2": row["l2_leaf_reg"],
            "best_it": row["best_iteration"],
            "val_rmse": row["val_rmse"],
            "val_mae": row["val_mae"],
            "val_r2": row["val_r2"],
        })

        if best_row is None or row["val_rmse"] < best_row["val_rmse"]:
            best_row = row
            best_model = model

    # tabela svih run-ova sortirana po val_rmse
    tuning_rows_sorted = sorted(tuning_rows, key=lambda r: r["val_rmse"])
    tuning_rows_print = [
        {
            **r,
            "val_rmse": fmt_float(r["val_rmse"], 4),
            "val_mae": fmt_float(r["val_mae"], 4),
            "val_r2": fmt_float(r["val_r2"], 4),
        }
        for r in tuning_rows_sorted
    ]

    print_table(
        tuning_rows_print,
        cols=["run", "depth", "lr", "l2", "best_it", "val_rmse", "val_mae", "val_r2"],
        title="=== CATBOOST TUNING SUMMARY (sorted by VAL_RMSE) ==="
    )

    # najbolji model
    model = best_model

    # final metrics za best model
    pred_val = model.predict(xva)
    val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

    pred_test = model.predict(xte)
    test_rmse, test_mae, test_r2 = eval_reg(y_test, pred_test)

    res = FitResult(
        name="CatBoost (tuned)",
        imputer=imputer,
        scaler="none",
        is_tuned=True,
        best_params=best_row,
        cv_rmse=None,
        val_rmse=val_rmse, val_mae=val_mae, val_r2=val_r2,
        test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2
    )

    print("\n=== CATBOOST BEST (by VAL) ===")
    print(
        f"depth={best_row['depth']}  lr={best_row['learning_rate']}  "
        f"l2={best_row['l2_leaf_reg']}  best_it={best_row['best_iteration']}"
    )
    print(f"VAL : RMSE={res.val_rmse:.4f}  MAE={res.val_mae:.4f}  R2={res.val_r2:.4f}")
    print(f"TEST: RMSE={res.test_rmse:.4f}  MAE={res.test_mae:.4f}  R2={res.test_r2:.4f}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # najbolji model
    out_path = out_dir / "results_catboost_tuned.csv"
    pd.DataFrame([res.__dict__]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # svi tuning run-ovi
    runs_path = out_dir / "catboost_tuning_runs.csv"
    pd.DataFrame(tuning_rows_sorted).to_csv(runs_path, index=False)
    print(f"Saved: {runs_path}")


if __name__ == "__main__":
    main()
