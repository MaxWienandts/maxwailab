import numpy as np
import pandas as pd

import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# METRICS
# ==========================================================
def compute_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def bootstrap_lightgbm_forward_selection(
    df,
    target,
    n_bootstrap,
    n_max_variables,
    metric_to_optimize,
    hyperparameters
):

    """
    Perform bootstrap-based forward feature selection using LightGBM.

    Methodology
    -----------
    • Bootstrap sampling with replacement.
    • True Out-of-Bag (OOB) validation.
    • Greedy forward selection.
    • Metric evaluated strictly on OOB samples.
    • Feature ranking stability can be analyzed across bootstraps.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset including target column.
    target : str
        Name of target column (binary classification expected).
    n_bootstrap : int
        Number of bootstrap resamples.
    n_max_variables : int
        Maximum number of variables to select.
    metric_to_optimize : str
        Metric key returned by compute_metrics.
    hyperparameters : dict
        LightGBM hyperparameters.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing:
        - "variables": selected features per bootstrap
        - one DataFrame per metric
          (rows = number of variables, columns = bootstrap iteration)
    """
    
    X_full = df.drop(columns=[target])
    y_full = df[target].astype(int)

    n_samples = len(df)

    # Estruturas finais
    results_metrics = {
        "auc_roc": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": []
    }

    results_variables = []

    for b in tqdm(range(n_bootstrap)):

        rng = np.random.default_rng(b)

        # ======================================================
        # BOOTSTRAP
        # ======================================================
        bootstrap_idx = rng.integers(0, n_samples, n_samples)

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bootstrap_idx] = False

        if oob_mask.sum() == 0:
            continue  # caso raríssimo

        X_train = X_full.iloc[bootstrap_idx]
        y_train = y_full.iloc[bootstrap_idx]

        X_val = X_full.iloc[oob_mask]
        y_val = y_full.iloc[oob_mask]

        # ======================================================
        # FORWARD SELECTION
        # ======================================================
        selected_features = []
        remaining_features = list(X_full.columns)

        metrics_history = []

        for _ in range(min(n_max_variables, len(remaining_features))):

            best_feature = None
            best_score = -np.inf

            for feature in remaining_features:

                trial_features = selected_features + [feature]

                model_params = hyperparameters.copy()
                model_params["random_state"] = b

                model = lgb.LGBMClassifier(**model_params)

                model.fit(X_train[trial_features], y_train)

                y_proba = model.predict_proba(X_val[trial_features])[:, 1]

                metrics = compute_metrics(y_val, y_proba)

                score = metrics[metric_to_optimize]

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_metrics = metrics

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)

            metrics_history.append(best_metrics)

        # Salva resultados do bootstrap
        results_variables.append(selected_features)

        for metric_name in results_metrics.keys():
            results_metrics[metric_name].append(
                [step[metric_name] for step in metrics_history]
            )

    # ======================================================
    # OUTPUT
    # ======================================================
    metrics_df = {
        metric: pd.DataFrame(values)
        for metric, values in results_metrics.items()
    }

    variables_df = pd.DataFrame(results_variables)

    # Transpose so each row is a variable and each column is a bootstrap
    dict_return = {
        "variables": variables_df,
        **metrics_df
    }
    for key in dict_return:
        dict_return[key] = dict_return[key].T
    return dict_return



def performance_forward_selection_boxplot(df_metric, metric_name):

    """
    Creates a boxplot showing the distribution of a performance metric 
    across different numbers of selected variables, for multiple bootstrap iterations.

    This visualization helps to analyze how model performance behaves 
    as the number of features increases/decreases, including stability 
    (spread across bootstraps) and potential overfitting/underfitting patterns.

    Parameters
    ----------
    df_metric : pd.DataFrame
        DataFrame where:
        - Index = number of variables (0, 1, 2, ...) 
        - Columns = different bootstrap iterations
        - Values = performance metric (e.g. AUC, F1, RMSE, etc.)
    metric_name : str
        Name of the metric being plotted (used in axis label and title)
    figsize : tuple, optional
        Figure size (width, height), by default (15, 6)
    title : str or None, optional
        Custom title for the plot. If None, a default title is generated.
    xlabel : str, optional
        Label for the x-axis, by default "Number of Variables"
    ylabel : str or None, optional
        Label for the y-axis. If None, uses metric_name capitalized.
    palette : str, optional
        Color palette name for seaborn, by default "viridis"

    Returns
    -------
    None
        Displays the plot (does not return the figure/axis)

    Examples
    --------
    >>> performance_boxplot(df_auc_results, "AUC", figsize=(16, 7), palette="magma")
    >>> performance_boxplot(df_rmse, "RMSE", title="Model RMSE vs Number of Features")
    """
    
    # Garante cópia
    df_aux = df_metric.copy()

    # Índice representa número de variáveis (step)
    df_aux = df_aux.reset_index()
    df_aux.rename(columns={"index": "n_variables"}, inplace=True)

    # Começar contagem em 1
    df_aux["n_variables"] += 1

    # Converte para formato long
    df_long = df_aux.melt(
        id_vars="n_variables",
        var_name="bootstrap",
        value_name=metric_name
    )

    # =============================
    # Plot
    # =============================
    plt.rcParams.update(plt.rcParamsDefault)
    sns.set(rc={'figure.figsize': (15, 6)})
    sns.set_style("darkgrid")

    meanprops = dict(color='black', linewidth=2)

    ax = sns.boxplot(
        data=df_long,
        x="n_variables",
        y=metric_name,
        showmeans=True,
        meanline=True,
        meanprops=meanprops
    )

    # Remove grid explicitly (extra safety)
    ax.grid(False)

    # Ensure white background
    ax.set_facecolor("white")
    plt.gcf().set_facecolor("white")
    
    ax.set_title("Performance by Number of Variables")

    plt.show()


def variable_frequency_forward_selection(df, n_bootstraps, figsize = (15, 15)):
    """
    Creates a heatmap showing the frequency (proportion) with which each variable
    was selected in models with different numbers of variables.
    """

    df_aux = df.copy()

    # Convert the data into counts per variable for each n_variables combination
    df_variables_heatmap = (
        df_aux.iloc[:, :-1]
        .apply(pd.Series.value_counts, axis=1)
        .fillna(0)
    )

    # Normalize by number of bootstraps → convert counts into proportions
    df_variables_heatmap = df_variables_heatmap / n_bootstraps

    # Cumulative frequency (forward selection effect)
    df_variables_heatmap = df_variables_heatmap.cumsum()

    # Remove zeros
    df_variables_heatmap = df_variables_heatmap.replace({0: np.nan})

    # Ensure counting starts at 1
    df_variables_heatmap["n_variables"] = range(1, len(df_variables_heatmap) + 1)

    # Set index
    df_variables_heatmap = df_variables_heatmap.set_index("n_variables")

    # -------------------------------------------------
    # ROW ORDERING ALGORITHM (greedy by column)
    # -------------------------------------------------
    df_order = df_variables_heatmap.T.copy()

    ordered_rows = []
    remaining_rows = list(df_order.index)

    for col in df_order.columns:

        if not remaining_rows:
            break

        # choose row with largest value in this column
        best_row = df_order.loc[remaining_rows, col].idxmax()

        ordered_rows.append(best_row)
        remaining_rows.remove(best_row)

    # append remaining rows if any
    ordered_rows.extend(remaining_rows)

    df_order = df_order.loc[ordered_rows]

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(figsize=figsize)

    sns.set(font_scale=0.7)

    with sns.axes_style("white"):
        ax = sns.heatmap(
            df_order,
            linewidths=0.2,
            annot=True,
            fmt=".1f",
            cmap="seismic",
            vmin=-1,
            vmax=1
        )

    plt.title("Most used variables")

    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()
    plt.close()



# Display the top-k variables using the same ordering used in the
# variable_frequency_forward_selection heatmap.
def _compute_forward_selection_order(df_variables_heatmap):
    """
    Internal function to compute the greedy ordering used in the heatmap.
    """

    df_order = df_variables_heatmap.T.copy()

    ordered_rows = []
    remaining_rows = list(df_order.index)

    for col in df_order.columns:

        if not remaining_rows:
            break

        best_row = df_order.loc[remaining_rows, col].idxmax()

        ordered_rows.append(best_row)
        remaining_rows.remove(best_row)

    ordered_rows.extend(remaining_rows)

    return ordered_rows

def top_k_forward_selection_variables_by_frequency_usage(df, n_bootstraps, k=10):
    """
    Return the top-k variables using the same ordering used in the
    variable_frequency_forward_selection heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        result_bootstrap["variables"]

    n_bootstraps : int
        Number of bootstrap iterations.

    k : int, default=10
        Number of variables to return.

    return_df : bool, default=False
        If True, returns a DataFrame with variable ranks.

    Returns
    -------
    list or pd.DataFrame
        List of top-k variables or a DataFrame with ranking.
    """

    df_aux = df.copy()

    df_variables_heatmap = (
        df_aux.iloc[:, :-1]
        .apply(pd.Series.value_counts, axis=1)
        .fillna(0)
    )

    df_variables_heatmap = df_variables_heatmap / n_bootstraps
    df_variables_heatmap = df_variables_heatmap.cumsum()
    df_variables_heatmap = df_variables_heatmap.replace({0: np.nan})

    df_variables_heatmap["n_variables"] = range(1, len(df_variables_heatmap) + 1)
    df_variables_heatmap = df_variables_heatmap.set_index("n_variables")

    ordered_rows = _compute_forward_selection_order(df_variables_heatmap)

    top_k = ordered_rows[:k]

    return top_k



# Returns the k variables that produced the best performancefor a given metric.
def top_k_variables_by_forward_selection_boxplot(result_bootstrap, k, metric):
    """
    Returns the k variables that produced the best performance
    for a given metric.

    Parameters
    ----------
    result_bootstrap : dict
        Dictionary containing:
        - "variables"
        - performance metrics (e.g. "auc_roc", "accuracy")

    k : int
        Number of variables in the model

    metric : str
        Performance metric key inside result_bootstrap

    Returns
    -------
    variables : list
        List with the k variables

    performance : float
        Best performance obtained
    """

    df_vars = result_bootstrap["variables"]
    df_metric = result_bootstrap[metric]

    row = k - 1

    # performance row
    perf_row = df_metric.loc[row]

    # maximum performance
    performance = perf_row.max()

    # bootstrap column where max occurs
    best_col = perf_row.idxmax()

    # corresponding variables
    variables = df_vars.loc[:row, best_col].tolist()

    return variables, performance



# Verify model performance removing or adding variables.
def bootstrap_model_variable_comparison_paired_lgbm(
    df_train,
    base_variables,
    target_col,
    df_val=None,
    start_month_col=None,
    variables_to_remove=None,
    variables_to_add=None,
    n_bootstrap=100,
    metric="auc",
    hyperparameters=None
):
    """
    Paired bootstrap comparison between two LightGBM classification models:
    
        1) Baseline (base_variables)
        2) Modified (remove/add variables)

    Uses stratified bootstrap.
    Returns formal statistical inference.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from lightgbm import LGBMClassifier
    from sklearn.metrics import (
        roc_auc_score,
        log_loss,
        accuracy_score,
        f1_score
    )

    if hyperparameters is None:
        hyperparameters = {}

    # -----------------------------------
    # Build modified variable list
    # -----------------------------------
    modified_variables = base_variables.copy()

    if variables_to_remove:
        modified_variables = [v for v in modified_variables if v not in variables_to_remove]

    if variables_to_add:
        modified_variables = list(set(modified_variables + variables_to_add))

    better_is_lower = metric in ["logloss"]

    # -----------------------------------
    # Data split
    # -----------------------------------
    if df_val is None:
    
        if start_month_col is not None:
            # Temporal split (time-ordered)
            df_train = df_train.sort_values(start_month_col)
            split_index = int(len(df_train) * 0.8)
    
            df_val = df_train.iloc[split_index:].copy()
            df_train = df_train.iloc[:split_index].copy()
    
        else:
            # Random stratified split (i.i.d. assumption)
            df_train, df_val = train_test_split(
                df_train,
                test_size=0.2,
                stratify=df_train[target_col],
                random_state=42
            )

    base_scores = []
    mod_scores = []
    differences = []

    # -----------------------------------
    # Stratified bootstrap
    # -----------------------------------
    for b in tqdm(range(n_bootstrap)):

        # ======================================================
        # BOOTSTRAP
        # ======================================================
        n_samples = len(df_train)
        rng = np.random.default_rng(b)
        bootstrap_idx = rng.integers(0, n_samples, n_samples)

        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bootstrap_idx] = False

        if oob_mask.sum() == 0:
            continue  # caso raríssimo

        df_boot = df_train.iloc[bootstrap_idx]

        if df_boot[target_col].nunique() < 2:
            continue

        # ---------------------------
        # Baseline model
        # ---------------------------
        model_base = LGBMClassifier(**hyperparameters)
        model_base.fit(
            df_boot[base_variables],
            df_boot[target_col]
        )

        preds_base = model_base.predict_proba(df_val[base_variables])[:, 1]

        # ---------------------------
        # Modified model
        # ---------------------------
        model_mod = LGBMClassifier(**hyperparameters)
        model_mod.fit(
            df_boot[modified_variables],
            df_boot[target_col]
        )

        preds_mod = model_mod.predict_proba(df_val[modified_variables])[:, 1]

        # ---------------------------
        # Metric computation
        # ---------------------------
        y_val = df_val[target_col]

        if metric == "auc":
            base_score = roc_auc_score(y_val, preds_base)
            mod_score = roc_auc_score(y_val, preds_mod)

        elif metric == "logloss":
            base_score = log_loss(y_val, preds_base)
            mod_score = log_loss(y_val, preds_mod)

        elif metric == "accuracy":
            base_score = accuracy_score(y_val, preds_base > 0.5)
            mod_score = accuracy_score(y_val, preds_mod > 0.5)

        elif metric == "f1":
            base_score = f1_score(y_val, preds_base > 0.5)
            mod_score = f1_score(y_val, preds_mod > 0.5)

        else:
            raise ValueError("Unsupported metric")

        base_scores.append(base_score)
        mod_scores.append(mod_score)
        differences.append(mod_score - base_score)

    base_scores = np.array(base_scores)
    mod_scores = np.array(mod_scores)
    differences = np.array(differences)

    # ======================================================
    # PLOT 1: Validation Performance Comparison
    # ======================================================

    removed_str = ", ".join(variables_to_remove) if variables_to_remove else "None"
    added_str = ", ".join(variables_to_add) if variables_to_add else "None"

    base_label = f"Baseline"
    mod_label = f"Modified\nRemoved: {removed_str}\nAdded: {added_str}"
    
    plt.figure(figsize=(8, 6))
    
    box = plt.boxplot(
        [base_scores, mod_scores],
        patch_artist=True,
        widths=0.6
    )
    
    # Color styling
    colors = ["#4C72B0", "#55A868"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Emphasize medians
    for median in box['medians']:
        median.set_color("black")
        median.set_linewidth(2)
    
    # Jittered points
    for i, scores in enumerate([base_scores, mod_scores], start=1):
        x = np.random.normal(i, 0.04, size=len(scores))
        plt.scatter(x, scores, alpha=0.4)
    
    plt.style.use("default")  # ensures white background
    plt.xticks([1, 2], [base_label, mod_label])
    plt.ylabel(metric.upper())
    plt.title("Validation Performance (Bootstrap Distribution)")
    
    plt.grid(False)  # removes grid completely
    
    plt.gcf().set_facecolor("white")   # figure background
    plt.gca().set_facecolor("white")   # axes background
    
    plt.tight_layout()
    plt.show()
    
    # ======================================================
    # PLOT 2: Paired Difference (Validation)
    # ======================================================
    # Determine direction
    if better_is_lower:
        direction_text = "Negative values favor Modified (lower is better)"
    else:
        direction_text = "Positive values favor Modified (higher is better)"
    plt.figure(figsize=(7, 5))
    
    box = plt.boxplot(
        differences,
        patch_artist=True,
        widths=0.5
    )
    
    box['boxes'][0].set_facecolor("#C44E52")
    box['boxes'][0].set_alpha(0.6)
    
    box['medians'][0].set_color("black")
    box['medians'][0].set_linewidth(2)
    
    # Jittered differences
    x = np.random.normal(1, 0.04, size=len(differences))
    plt.scatter(x, differences, alpha=0.4)
    
    plt.axhline(0, linestyle="--")
    plt.xticks([1], ["Modified − Baseline"])
    plt.ylabel(metric.upper())
    plt.title(
        "Paired Validation Difference (Modified − Baseline)\n"
        f"{direction_text}"
    )
    plt.grid(False)  # removes grid completely
    plt.tight_layout()
    plt.show()

    # ======================================================
    # Inference
    # ======================================================
    mean_diff = np.mean(differences)
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])

    if better_is_lower:
        prob_better = np.mean(differences < 0)
    else:
        prob_better = np.mean(differences > 0)

    return {
        "baseline_vars": base_variables,
        "baseline_val_mean": np.mean(base_scores),
        "modified_val_mean": np.mean(mod_scores),
        "mean_difference_val": mean_diff,
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "probability_modified_better": prob_better,
        "n_effective_bootstrap": len(differences)
    }