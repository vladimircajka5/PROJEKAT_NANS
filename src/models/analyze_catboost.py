import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

    # koristi iste izbore kao u tuningu
    imputer = "median"
    best = {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10.0}

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
        iterations=8000,
        learning_rate=best["learning_rate"],
        depth=best["depth"],
        l2_leaf_reg=best["l2_leaf_reg"],
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

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    fi = model.get_feature_importance()
    fi_df = pd.DataFrame({
        "feature": xtr.columns,
        "importance": fi
    }).sort_values("importance", ascending=False)

    fi_path = out_dir / "catboost_feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Saved: {fi_path}")

    topk = 20
    top = fi_df.head(topk).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"])
    plt.title("CatBoost feature importance (top 20)")
    plt.xlabel("importance")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_feature_importance.png", dpi=160)
    plt.close()
    print(f"Saved: {out_dir / 'catboost_feature_importance.png'}")

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
        n_jobs=1
    )

    perm_df = pd.DataFrame({
        "feature": xva.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    perm_path = out_dir / "catboost_permutation_importance_val.csv"
    perm_df.to_csv(perm_path, index=False)
    print(f"Saved: {perm_path}")

    top_perm = perm_df.head(topk).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_perm["feature"], top_perm["importance_mean"])
    plt.title("Permutation importance on VAL (top 20, negative RMSE)")
    plt.xlabel("importance (increase in error)")
    plt.tight_layout()
    plt.savefig(out_dir / "catboost_permutation_importance_val.png", dpi=160)
    plt.close()
    print(f"Saved: {out_dir / 'catboost_permutation_importance_val.png'}")


if __name__ == "__main__":
    main()
