from pathlib import Path

import pandas as pd

RESULT_FILES = [
    ("Linear baselines", "results_linear_baselines.csv"),
    ("CatBoost (tuned)", "results_catboost_tuned.csv"),
]

CSV_DIR = Path("results/csv")

SHOW_COLS = [
    "name", "imputer", "scaler", "is_tuned",
    "val_rmse", "val_mae", "val_r2",
    "test_rmse", "test_mae", "test_r2",
]

SORT_BY = ["val_rmse", "test_rmse"]


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)

    for c in ["best_params", "cv_rmse", "is_tuned", "imputer", "scaler", "name"]:
        if c not in df.columns:
            df[c] = None

    for c in ["val_rmse", "val_mae", "val_r2", "test_rmse", "test_mae", "test_r2", "cv_rmse"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def fmt_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.copy()

    for c in ["val_rmse", "val_mae", "test_rmse", "test_mae", "cv_rmse"]:
        if c in view.columns:
            view[c] = view[c].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    for c in ["val_r2", "test_r2"]:
        if c in view.columns:
            view[c] = view[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

    for c in SHOW_COLS:
        if c not in view.columns:
            view[c] = ""

    view = view[SHOW_COLS].head(max_rows)

    return view.to_string(index=False)


def best_per_model(df_all: pd.DataFrame) -> pd.DataFrame:
    d = df_all.sort_values(SORT_BY).copy()
    return d.groupby("name", as_index=False).head(1).sort_values(SORT_BY)

def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    loaded = []
    for _, fname in RESULT_FILES:
        p = CSV_DIR / fname
        df = read_csv_safe(p)
        if df is None:
            print(f"Warning: {p} not found, skipping.")
            continue
        df["source_file"] = fname
        loaded.append(df)

    if not loaded:
        print("\nNo result CSVs found in ./results. Nothing to summarize.")
        return

    df_all = pd.concat(loaded, ignore_index=True)

    df_all = df_all.sort_values(SORT_BY).reset_index(drop=True)

    print("\n=== OVERALL RANKING ===")
    print(fmt_table(df_all, max_rows=20))

    best = df_all.iloc[0].to_dict()
    print("\n=== BEST OVERALL ===")
    print(
        f"{best.get('name')} | imputer={best.get('imputer')} scaler={best.get('scaler')} tuned={best.get('is_tuned')}\n"
        f"VAL : RMSE={best.get('val_rmse'):.4f}  MAE={best.get('val_mae'):.4f}  R2={best.get('val_r2'):.4f}\n"
        f"TEST: RMSE={best.get('test_rmse'):.4f}  MAE={best.get('test_mae'):.4f}  R2={best.get('test_r2'):.4f}"
    )

    per_model = best_per_model(df_all)
    print("\n=== BEST PER MODEL ===")
    print(fmt_table(per_model, max_rows=50))

    merged_path = CSV_DIR / "summarized_results.csv"
    df_all.to_csv(merged_path, index=False)

if __name__ == "__main__":
    main()
