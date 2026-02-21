from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RESULT_FILES = [
    ("Linear baselines", "results_linear_baselines.csv"),
    ("CatBoost (tuned)", "results_catboost_tuned.csv"),
]

CSV_DIR = Path("results/csv")
PLOT_DIR = Path("results/plots")

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

def ensure_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

def save_plots(df_all: pd.DataFrame, per_model: pd.DataFrame):
    # najbolji TEST RMSE po modelu
    if "test_rmse" in per_model.columns and per_model["test_rmse"].notna().any():
        plt.figure(figsize=(9, 5))
        plt.bar(per_model["name"], per_model["test_rmse"])
        plt.ylabel("TEST RMSE")
        plt.title("Best TEST RMSE per model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out = PLOT_DIR / "best_test_rmse_per_model.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved plot: {out}")

    #  VAL vs TEST RMSE (najbolji po modelu)
    if (
        "val_rmse" in per_model.columns and "test_rmse" in per_model.columns
        and per_model["val_rmse"].notna().any() and per_model["test_rmse"].notna().any()
    ):
        plt.figure(figsize=(6, 6))
        plt.scatter(per_model["val_rmse"], per_model["test_rmse"], alpha=0.8)
        for _, r in per_model.iterrows():
            if pd.notna(r["val_rmse"]) and pd.notna(r["test_rmse"]):
                plt.text(r["val_rmse"], r["test_rmse"], str(r["name"]), fontsize=8)
        plt.xlabel("VAL RMSE")
        plt.ylabel("TEST RMSE")
        plt.title("VAL vs TEST RMSE (best per model)")
        plt.tight_layout()
        out = PLOT_DIR / "val_vs_test_rmse_best_per_model.png"
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved plot: {out}")

    # najbolji TEST RMSE po CSV fajlu
    if "source_file" in df_all.columns and "test_rmse" in df_all.columns:
        best_by_source = (
            df_all.sort_values(SORT_BY)
                 .groupby("source_file", as_index=False)
                 .head(1)
                 .sort_values(SORT_BY)
        )
        if best_by_source["test_rmse"].notna().any():
            plt.figure(figsize=(8, 5))
            plt.bar(best_by_source["source_file"], best_by_source["test_rmse"])
            plt.ylabel("TEST RMSE")
            plt.title("Best TEST RMSE per result file")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            out = PLOT_DIR / "best_test_rmse_per_file.png"
            plt.savefig(out, dpi=160)
            plt.close()
            print(f"Saved plot: {out}")

def main():
    ensure_dirs()

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

    # Sacuvaj CSV-ove sa svim rezultatima i najboljim po modelu
    leaderboard_path = CSV_DIR / "leaderboard_all.csv"
    df_all.to_csv(leaderboard_path, index=False)
    print(f"Saved: {leaderboard_path}")

    best_path = CSV_DIR / "best_per_model.csv"
    per_model.to_csv(best_path, index=False)
    print(f"Saved: {best_path}")

    # Sacuvaj plotove
    save_plots(df_all, per_model)

if __name__ == "__main__":
    main()
