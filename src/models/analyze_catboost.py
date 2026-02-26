import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import ast

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from catboost import CatBoostRegressor

from src.data.make_dataset import load_dataset, prepare_dataframe, TARGET
from src.data.preprocess import build_preprocessor_catboost

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


def main():
    df = load_dataset(DATA_PATH)
    df = prepare_dataframe(df)

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    x_train, x_val, x_test, y_train, y_val, y_test = split(x, y)

    # koristi CatBoost BASELINE iz results/csv/results_catboost.csv
    row = pd.read_csv("results/csv/results_catboost.csv").iloc[0].to_dict()
    imputer = row.get("imputer", "median")

    best_raw = row.get("best_params", "{}")
    best = ast.literal_eval(best_raw) if isinstance(best_raw, str) else (best_raw or {})

    print("\n=== Analyzing CatBoost BASELINE ===")
    print("imputer:", imputer)
    print("best_params:", best)

    pre = build_preprocessor_catboost(df, imputer=imputer, missing_threshold=0.80)
    pre.fit(x_train, y_train)

    xtr = pre.transform(x_train)
    xva = pre.transform(x_val)

    cat_cols = xtr.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_features = [xtr.columns.get_loc(c) for c in cat_cols]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        iterations=int(best.get("iterations", 8000)),
        learning_rate=float(best.get("learning_rate", 0.05)),
        depth=int(best.get("depth", 8)),
        l2_leaf_reg=float(best.get("l2_leaf_reg", 3.0)),
        subsample=float(best.get("subsample", 0.8)),
        colsample_bylevel=float(best.get("colsample_bylevel", 0.8)),
        early_stopping_rounds=400,   # neka bude isto kao u baseline treningu
        verbose=200,
        allow_writing_files=False,
        thread_count=-1
    )

    model.fit(
        xtr, y_train,
        eval_set=(xva, y_val),
        cat_features=cat_features,
        use_best_model=True
    )

    csv_dir = Path("results/csv")
    plot_dir = Path("results/plots")
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    fi = model.get_feature_importance()
    fi_df = pd.DataFrame({
        "feature": xtr.columns,
        "importance": fi
    }).sort_values("importance", ascending=False)

    fi_path = csv_dir / "catboost_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved: {fi_path}")

    topk = 20
    top = fi_df.head(topk).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("CatBoost feature importance (top 20)")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(plot_dir / "catboost_feature_importance.png", dpi=160)
    plt.close()
    print(f"Saved: {plot_dir / 'catboost_feature_importance.png'}")

    # negativni RMSE (sklearn ocekuje da veca vrijednost bude bolja, a mi zelimo da manja RMSE bude bolja)
    def neg_rmse(est, x, y):
        pred = est.predict(x)
        rmse = np.sqrt(mean_squared_error(y, pred))
        return -rmse

    perm = permutation_importance(
        model, xva, y_val,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=neg_rmse,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": xva.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_path = csv_dir / "catboost_permutation_importance_val.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Saved: {perm_path}")

    top_perm = perm_df.head(topk).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_perm["feature"], top_perm["importance_mean"])
    plt.title("Permutation importance on VAL (top 20, negative RMSE)")
    plt.xlabel("importance (decrease in -RMSE)")
    plt.tight_layout()
    plt.savefig(plot_dir / "catboost_permutation_importance_val.png", dpi=160)
    plt.close()
    print(f"Saved: {plot_dir / 'catboost_permutation_importance_val.png'}")

    print("\n=== SHAP ANALYSIS ===")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(xva) 

    # SHAP summary plot (bee-swarm)
    plt.figure()
    shap.summary_plot(shap_values, xva, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_dir / 'shap_summary_beeswarm.png'}")

    # SHAP summary bar plot (globalne srednje |SHAP| vrednosti)
    plt.figure()
    shap.summary_plot(shap_values, xva, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_dir / 'shap_summary_bar.png'}")

    # Globalne SHAP vrednosti (CSV)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_global = pd.DataFrame({
        "feature": xva.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    shap_global.to_csv(csv_dir / "shap_global_importance.csv", index=False)
    print(f"Saved: {csv_dir / 'shap_global_importance.csv'}")

    # Dependence plot za top 3 atributa
    top3_features = shap_global["feature"].head(3).tolist()
    for feat in top3_features:
        plt.figure()
        shap.dependence_plot(feat, shap_values.values, xva, show=False)
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        plt.savefig(plot_dir / f"shap_dependence_{safe_name}.png",
                    dpi=160, bbox_inches="tight")
        plt.close()
    print(f"Saved: dependence plots for {top3_features}")

    # Lokalna objasnjenja (waterfall) za 3 primera:
    # 1. zajednica sa najvecom predikcijom
    # 2. zajednica sa najmanjom predikcijom
    # 3. zajednica blizu medijane
    pred_val = model.predict(xva)
    idx_high = int(np.argmax(pred_val))
    idx_low  = int(np.argmin(pred_val))
    idx_med  = int(np.argsort(pred_val)[len(pred_val) // 2])

    for label, idx in [("high", idx_high), ("low", idx_low), ("median", idx_med)]:
        plt.figure()
        shap.waterfall_plot(shap_values[idx], show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(plot_dir / f"shap_waterfall_{label}.png",
                    dpi=160, bbox_inches="tight")
        plt.close()
    print(f"Saved: waterfall plots (high/low/median predictions)")

    print("\n=== PREDICTION & RESIDUAL ANALYSIS ===")

    xte = pre.transform(x_test)
    pred_test = model.predict(xte)

    for split_name, y_true, y_pred in [("val", y_val, pred_val),
                                        ("test", y_test, pred_test)]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        residuals = np.array(y_true) - np.array(y_pred)

        # y_true vs y_pred scatter
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, s=18, edgecolors="k", linewidths=0.3)
        mn = min(float(np.min(y_true)), float(np.min(y_pred)))
        mx = max(float(np.max(y_true)), float(np.max(y_pred)))
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax.set_xlabel("y_true (violentPerPop)")
        ax.set_ylabel("y_pred")
        ax.set_title(f"CatBoost {split_name.upper()}: y_true vs y_pred\n"
                     f"RMSE={rmse:.1f}  MAE={mae:.1f}  R²={r2:.3f}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"catboost_pred_vs_true_{split_name}.png", dpi=160)
        plt.close(fig)

        # Residual plot (residuals vs y_pred)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.5, s=18, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("y_pred")
        ax.set_ylabel("Residual (y_true − y_pred)")
        ax.set_title(f"CatBoost {split_name.upper()}: Residuals vs Predicted")
        fig.tight_layout()
        fig.savefig(plot_dir / f"catboost_residuals_{split_name}.png", dpi=160)
        plt.close(fig)

        # Histogram reziduala
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residuals, bins=40, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"CatBoost {split_name.upper()}: Distribution of Residuals")
        fig.tight_layout()
        fig.savefig(plot_dir / f"catboost_residual_hist_{split_name}.png", dpi=160)
        plt.close(fig)

        print(f"Saved: pred_vs_true, residuals, residual_hist for {split_name}")

    print("\nDone – all analysis saved to results/csv/ and results/plots/")


if __name__ == "__main__":
    main()
