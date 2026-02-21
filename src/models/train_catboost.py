import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
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

def catboost_kfold_cv(df, x_train, y_train, imputer, params, folds=3,
                      iterations=2000, early_stopping_rounds=80):
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    rmses, maes, r2s = [], [], []

    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    for tr_idx, va_idx in cv.split(x_train):
        x_tr, y_tr = x_train.iloc[tr_idx], y_train.iloc[tr_idx]
        x_va, y_va = x_train.iloc[va_idx], y_train.iloc[va_idx]

        pre = build_preprocessor_catboost(df, imputer=imputer, missing_threshold=0.80)
        pre.fit(x_tr, y_tr)

        Xtr = pre.transform(x_tr)
        Xva = pre.transform(x_va)

        cat_cols = Xtr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        cat_features = [Xtr.columns.get_loc(c) for c in cat_cols]

        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=RANDOM_STATE,
            iterations=iterations,
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            subsample=0.8,
            colsample_bylevel=0.8,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

        model.fit(Xtr, y_tr, eval_set=(Xva, y_va),
                  cat_features=cat_features, use_best_model=True)

        pred = model.predict(Xva)
        rmse, mae, r2 = eval_reg(y_va, pred)

        rmses.append(rmse); maes.append(mae); r2s.append(r2)

    rmse_mean = float(np.mean(rmses))
    rmse_std  = float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0

    return {"cv_rmse": rmse_mean, "cv_rmse_std": rmse_std,
            "cv_mae": float(np.mean(maes)), "cv_r2": float(np.mean(r2s))}

def main():
    df = load_dataset(DATA_PATH)
    df = prepare_dataframe(df)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

    imputer = "median"

    # tuning
    grid = [
        {"depth": 6,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.03, "l2_leaf_reg": 3.0},
        {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 10.0},
    ]

    best_row = None
    tuning_rows = []

    print("\n=== CATBOOST TUNING (3-FOLD CV on TRAIN) ===")
    for i, g in enumerate(grid, start=1):
        cv_stats = catboost_kfold_cv(
            df,
            x_train, y_train,
            imputer=imputer,
            params=g,
            folds=3,
            iterations=2000,
            early_stopping_rounds=80,
        )

        row = {**g, **cv_stats}

        print(f"[{i}/{len(grid)}] depth={g['depth']} lr={g['learning_rate']} "
              f"l2={g['l2_leaf_reg']} | CV_RMSE={row['cv_rmse']:.4f}±{row['cv_rmse_std']:.4f}")

        tuning_rows.append({
            "run": i,
            "depth": g["depth"],
            "lr": g["learning_rate"],
            "l2": g["l2_leaf_reg"],
            "cv_rmse": row["cv_rmse"],
            "cv_rmse_std": row["cv_rmse_std"],
            "cv_mae": row["cv_mae"],
            "cv_r2": row["cv_r2"],
        })

        if best_row is None or row["cv_rmse"] < best_row["cv_rmse"]:
            best_row = row

    # tabela svih run-ova sortirana po CV RMSE
    tuning_rows_sorted = sorted(tuning_rows, key=lambda r: r["cv_rmse"])
    tuning_rows_print = [
        {
            **r,
            "cv_rmse": fmt_float(r["cv_rmse"], 4),
            "cv_rmse_std": fmt_float(r["cv_rmse_std"], 4),
            "cv_mae": fmt_float(r["cv_mae"], 4),
            "cv_r2": fmt_float(r["cv_r2"], 4),
        }
        for r in tuning_rows_sorted
    ]

    print_table(
        tuning_rows_print,
        cols=["run", "depth", "lr", "l2", "cv_rmse", "cv_rmse_std", "cv_mae", "cv_r2"],
        title="=== CATBOOST TUNING SUMMARY (sorted by CV_RMSE) ==="
    )

    print(f"\nBest params: depth={best_row['depth']} lr={best_row['learning_rate']} "
          f"l2={best_row['l2_leaf_reg']}")
    print(f"CV : RMSE={best_row['cv_rmse']:.4f} ± {best_row['cv_rmse_std']:.4f}  "
          f"MAE={best_row['cv_mae']:.4f}  R2={best_row['cv_r2']:.4f}")

    # finalni model sa najboljim parametrima (fit na TRAIN, early stop na VAL) 
    pre = build_preprocessor_catboost(df, imputer=imputer, missing_threshold=0.80)
    pre.fit(x_train, y_train)
    xtr = pre.transform(x_train)
    xva = pre.transform(x_val)
    xte = pre.transform(x_test)

    cat_cols = xtr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_features = [xtr.columns.get_loc(c) for c in cat_cols]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        iterations=8000,
        learning_rate=best_row["learning_rate"],
        depth=best_row["depth"],
        l2_leaf_reg=best_row["l2_leaf_reg"],
        subsample=0.8,
        colsample_bylevel=0.8,
        early_stopping_rounds=200,
        verbose=200,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(xtr, y_train, eval_set=(xva, y_val),
              cat_features=cat_features, use_best_model=True)

    pred_val = model.predict(xva)
    val_rmse, val_mae, val_r2 = eval_reg(y_val, pred_val)

    pred_test = model.predict(xte)
    test_rmse, test_mae, test_r2 = eval_reg(y_test, pred_test)

    best_it = int(model.get_best_iteration()) if model.get_best_iteration() is not None else None

    res = FitResult(
        name="CatBoost (tuned)",
        imputer=imputer,
        scaler="none",
        is_tuned=True,
        best_params={"depth": best_row["depth"],
        "learning_rate": best_row["learning_rate"],
        "l2_leaf_reg": best_row["l2_leaf_reg"],
        "cv_folds": 3,
        "cv_rmse_std": best_row["cv_rmse_std"],
        "cv_mae": best_row["cv_mae"],
        "cv_r2": best_row["cv_r2"],
        "best_iteration": best_it
        },
        cv_rmse=best_row["cv_rmse"],
        val_rmse=val_rmse, val_mae=val_mae, val_r2=val_r2,
        test_rmse=test_rmse, test_mae=test_mae, test_r2=test_r2
    )

    print("\n=== CATBOOST BEST (by CV) ===")
    print(f"depth={best_row['depth']}  lr={best_row['learning_rate']}  "
          f"l2={best_row['l2_leaf_reg']}  best_it={best_it}")
    print(f"VAL : RMSE={val_rmse:.4f}  MAE={val_mae:.4f}  R2={val_r2:.4f}")
    print(f"TEST: RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  R2={test_r2:.4f}")

    csv_dir = Path("results/csv")
    csv_dir.mkdir(parents=True, exist_ok=True)

    # najbolji model
    out_path = csv_dir / "results_catboost_tuned.csv"
    pd.DataFrame([res.__dict__]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # svi tuning run-ovi
    runs_path = csv_dir / "catboost_tuning_runs.csv"
    pd.DataFrame(tuning_rows_sorted).to_csv(runs_path, index=False)
    print(f"Saved: {runs_path}")


if __name__ == "__main__":
    main()
