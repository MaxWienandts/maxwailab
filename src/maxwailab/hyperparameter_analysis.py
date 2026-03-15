import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def lightgbm_hyperparameter_auc_curve_bootstrap(
    X_train,
    y_train,
    X_val,
    y_val,
    hyperparameters,
    hyperparameter_name,
    hyperparameter_values,
    n_bootstrap=100,
    ci=0.95,
    random_state=42
):
    """
    Hyperparameter sensitivity analysis with bootstrap confidence intervals.

    Bootstrap is applied to the training set.
    Validation set is fixed (out-of-time).
    """

    rng = np.random.default_rng(random_state)

    results = []

    n = len(X_train)

    for value in hyperparameter_values:

        params = hyperparameters.copy()
        params[hyperparameter_name] = value
        params["random_state"] = random_state

        bootstrap_scores = []

        # ----------------------
        # Bootstrap training
        # ----------------------

        for _ in range(n_bootstrap):

            idx = rng.integers(0, n, n)

            X_boot = X_train.iloc[idx]
            y_boot = y_train.iloc[idx]

            model = lgb.LGBMClassifier(**params)

            model.fit(X_boot, y_boot)

            pred = model.predict_proba(X_boot)[:, 1]

            auc = roc_auc_score(y_boot, pred)

            bootstrap_scores.append(auc)

        bootstrap_scores = np.array(bootstrap_scores)

        alpha = (1 - ci) / 2

        ci_lower = np.quantile(bootstrap_scores, alpha)
        ci_upper = np.quantile(bootstrap_scores, 1 - alpha)

        # ----------------------
        # Train final model
        # ----------------------

        model = lgb.LGBMClassifier(**params)

        model.fit(X_train, y_train)

        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]

        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)

        results.append({
            hyperparameter_name: value,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "bootstrap_mean": bootstrap_scores.mean(),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        })

    df_results = pd.DataFrame(results)

    # ----------------------------------
    # Select best hyperparameter value
    # ----------------------------------
    
    best_idx = df_results["val_auc"].idxmax()
    best_value = df_results.loc[best_idx, hyperparameter_name]
    
    best_hyperparameters = hyperparameters.copy()
    best_hyperparameters[hyperparameter_name] = best_value

    # ----------------------
    # Plot
    # ----------------------

    x = df_results[hyperparameter_name]

    plt.figure(figsize=(8, 5))

    plt.plot(x, df_results["train_auc"], marker="o", label="Train AUC")
    plt.plot(x, df_results["val_auc"], marker="o", label="Validation AUC")

    plt.fill_between(
        x,
        df_results["ci_lower"],
        df_results["ci_upper"],
        alpha=0.25,
        label="Bootstrap CI (train)"
    )

    plt.xlabel(hyperparameter_name)
    plt.ylabel("ROC AUC")

    plt.title("Hyperparameter Sensitivity Curve")

    plt.grid(True)
    plt.legend()

    plt.show()

    return {
        "results": df_results,
        "best_hyperparameters": best_hyperparameters
    }