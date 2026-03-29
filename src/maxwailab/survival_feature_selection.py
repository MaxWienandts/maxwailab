import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# BOOTSTRAP
# ==========================================================
def discrete_duration_bootstrap(
    df,
    duration_col,
    event_col,
    rng
):
    """
    Bootstrap preserving:
        - exact discrete duration distribution
        - exact event proportion within each duration
    """

    df = df.copy()

    # Stratify by (duration, event)
    grouped = df.groupby([duration_col, event_col], group_keys=False)

    boot_samples = []

    for (duration, event), group in grouped:
        n = len(group)

        # Sample with replacement within this exact stratum
        idx = rng.integers(0, n, n)
        boot_samples.append(group.iloc[idx])

    df_boot = pd.concat(boot_samples).reset_index(drop=True)

    return df_boot

# ==========================================================
# METRICS
# ==========================================================
def compute_survival_metrics(
    model,
    df_train,
    df_val,
    duration_col,
    event_col,
    n_times=100
):
    """
    Compute survival metrics for lifelines models.

    Returns:
        dict with:
            - Harrell C-index: ranking discrimination
            - c_index_ipcw: censoring-adjusted concordance
            - Integrated Brier Score: overall calibration + discrimination
            - Mean Time-dependent Brier Score
    """

    # =====================================================
    # Structured arrays (required by sksurv)
    # =====================================================

    y_train_struct = Surv.from_dataframe(event_col, duration_col, df_train)
    y_val_struct = Surv.from_dataframe(event_col, duration_col, df_val)

    # =====================================================
    # Risk scores (for C-index)
    # =====================================================

    if hasattr(model, "predict_partial_hazard"):
        # Cox models
        risk_scores = model.predict_partial_hazard(df_val).values.ravel()
    else:
        # AFT models → higher survival time = lower risk
        risk_scores = -model.predict_expectation(df_val).values.ravel()

    # Standard C-index
    c_index = concordance_index(
        df_val[duration_col],
        -risk_scores,  # lifelines expects higher survival = better
        df_val[event_col]
    )

    # IPCW C-index
    c_ipcw = concordance_index_ipcw(
        y_train_struct,
        y_val_struct,
        risk_scores
    )[0]

    # =====================================================
    # Time grid (STRICTLY inside validation follow-up)
    # =====================================================

    t_min = df_val[duration_col].min()
    t_max = df_val[duration_col].max()

    # Must satisfy: t < max(T_val)
    times_eval = np.linspace(t_min, t_max - 1e-8, n_times)

    # =====================================================
    # Survival probability matrix
    # =====================================================

    surv_df = model.predict_survival_function(df_val)
    
    # Interpolate survival curves at evaluation times
    surv_interp = (
        surv_df
        .reindex(surv_df.index.union(times_eval))
        .interpolate(method="index")
        .loc[times_eval]
    )

    # shape → (n_samples, n_times)
    surv_preds = surv_interp.T.values
    if np.isnan(surv_preds).any():
        return {
            "c_index": np.nan,
            "c_index_ipcw": np.nan,
            "ibs": np.inf,
            "mean_brier": np.inf
        }
    
    # =====================================================
    # Brier score + IBS
    # =====================================================

    _, bs_values = brier_score(
        y_train_struct,
        y_val_struct,
        surv_preds,
        times_eval
    )

    ibs = integrated_brier_score(
        y_train_struct,
        y_val_struct,
        surv_preds,
        times_eval
    )

    return {
        "c_index": c_index,
        "c_index_ipcw": c_ipcw,
        "ibs": ibs,
        "mean_brier": np.mean(bs_values)
    }
    
    
# ==========================================================
# MAIN FUNCTION
# ==========================================================
def bootstrap_survival_forward_selection(
    df_train,
    duration_col,
    event_col,
    start_month_col,
    model,
    n_bootstrap,
    n_max_variables,
    metric_to_optimize,
    hyperparameters,
    df_val=None
):
    """
    Bootstrap forward selection for cohort survival data.

    If df_val is None:
        - Uses last 20% most recent start_month as validation.

    Bootstrap is applied ONLY to training set.
    """

    # ------------------------------------------------------
    # Temporal split if df_val not provided
    # ------------------------------------------------------
    if df_val is None:

        df_train = df_train.sort_values(start_month_col)

        split_index = int(len(df_train) * 0.8)

        df_val = df_train.iloc[split_index:].copy()
        df_train = df_train.iloc[:split_index].copy()

    # ------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------
    feature_cols = [
        col for col in df_train.columns
        if col not in [duration_col, event_col, start_month_col]
    ]

    results_metrics = {
        "c_index": [],
        "c_index_ipcw": [],
        "ibs": [],
        "mean_brier": []
    }

    results_variables = []

    n_train = len(df_train)

    # ------------------------------------------------------
    # Bootstrap Loop
    # ------------------------------------------------------
    for b in tqdm(range(n_bootstrap)):

        rng = np.random.default_rng(b)

        # IID bootstrap on training users
        df_train_boot = discrete_duration_bootstrap(
            df_train,
            duration_col=duration_col,
            event_col=event_col,
            rng=rng
        )
        if df_train_boot[event_col].sum() < 10:
            continue  # skip bootstrap draw
        
        selected_features = []
        remaining_features = feature_cols.copy()
        metrics_history = []

 
        # --------------------------------------------------
        # Forward Selection
        # --------------------------------------------------
        for _ in range(min(n_max_variables, len(remaining_features))):

            best_feature = None
            best_score = -np.inf

            for feature in remaining_features:

                trial_features = selected_features + [feature]

                model = model.__class__(**model._kwargs) \
                    if hasattr(model, "_kwargs") else model.__class__()

                model.fit(
                    df_train_boot[trial_features + [duration_col, event_col]],
                    duration_col=duration_col,
                    event_col=event_col
                )

                metrics = compute_survival_metrics(
                    model,
                    df_train_boot[trial_features + [duration_col, event_col]],
                    df_val[trial_features + [duration_col, event_col]],
                    duration_col,
                    event_col,
                )

                score = metrics[metric_to_optimize]

                # Reverse sign for minimization metrics
                if metric_to_optimize in ["ibs", "mean_brier"]:
                    score = -score

                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_metrics = metrics

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            metrics_history.append(best_metrics)

        results_variables.append(selected_features)

        for metric_name in results_metrics:
            results_metrics[metric_name].append(
                [step[metric_name] for step in metrics_history]
            )

    return {
        "variables": pd.DataFrame(results_variables).T,
        **{
            k: pd.DataFrame(v).T
            for k, v in results_metrics.items()
        }
    }


def bootstrap_model_variable_comparison_paired(
    df_train,
    model,
    base_variables,
    df_val=None,
    duration_col='duration',
    event_col='event',
    start_month_col='start_month',
    variables_to_remove=None,
    variables_to_add=None,
    n_bootstrap=50,
    metric="ibs",
    hyperparameters=None
):

    """
    Performs a paired bootstrap comparison between two survival models:
    
        1) Baseline model using `base_variables`
        2) Modified model obtained by removing and/or adding variables
    
    The function:
    - Uses temporal split (80/20) if `df_val` is not provided.
    - Applies a stratified discrete bootstrap on the training set
      preserving the duration distribution.
    - Fits both models on each bootstrap sample.
    - Evaluates validation performance using the selected survival metric
      (e.g., IBS, mean_brier, c_index, c_index_ipcw).
    - Produces two plots:
        • Validation performance distributions (baseline vs modified)
        • Paired bootstrap difference distribution (Modified − Baseline)
          with interpretation of improvement direction.
    - Returns statistical inference including:
        • Mean validation performance (baseline and modified)
        • Mean paired difference
        • 95% bootstrap confidence interval
        • Probability that modified model outperforms baseline
        • Effective number of bootstrap samples used
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataset.
    
    model : model instance.
    
    base_variables : list[str]
        Variables used in the baseline model.
    
    df_val : pd.DataFrame or None
        Validation dataset. If None, a temporal 80/20 split is applied.
    
    duration_col : str
        Column containing survival times.
    
    event_col : str
        Column containing event indicator (1=event, 0=censored).
    
    start_month_col : str
        Column used for temporal ordering when splitting.
    
    variables_to_remove : list[str] or None
        Variables removed from baseline in modified model.
    
    variables_to_add : list[str] or None
        Variables added to baseline in modified model.
    
    n_bootstrap : int
        Number of bootstrap iterations.
    
    metric : str
        Performance metric to evaluate.
    
    Returns
    -------
    dict
        Statistical summary of paired bootstrap comparison.
    """
    
    if hyperparameters is None:
        hyperparameters = {}

    # -------------------------------
    # Build modified variable list
    # -------------------------------
    modified_variables = base_variables.copy()

    if variables_to_remove is not None:
        modified_variables = [
            v for v in modified_variables if v not in variables_to_remove
        ]

    if variables_to_add is not None:
        modified_variables = list(set(modified_variables + variables_to_add))

    better_is_lower = metric in ["ibs", "mean_brier"]

    # ------------------------------------------------------
    # Temporal split if df_val not provided
    # ------------------------------------------------------
    if df_val is None:

        df_train = df_train.sort_values(start_month_col)
        split_index = int(len(df_train) * 0.8)

        df_val = df_train.iloc[split_index:].copy()
        df_train = df_train.iloc[:split_index].copy()

    # Storage
    base_val_scores = []
    mod_val_scores = []
    differences = []

    # ======================================================
    # Bootstrap loop
    # ======================================================
    for b in tqdm(range(n_bootstrap)):

        rng = np.random.default_rng(b)

        df_train_boot = discrete_duration_bootstrap(
            df_train,
            duration_col,
            event_col,
            rng
        )

        # Skip degenerate bootstrap draws
        if df_train_boot[event_col].sum() < 5:
            continue

        # ==========================
        # BASELINE MODEL
        # ==========================      
        model_base = model.__class__(**model._kwargs) \
            if hasattr(model, "_kwargs") else model.__class__()

        model_base.fit(
            df_train_boot[base_variables  + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col
        )

        metrics_base_val = compute_survival_metrics(
            model_base,
            df_train_boot[base_variables + [duration_col, event_col]],
            df_val[base_variables + [duration_col, event_col]],
            duration_col,
            event_col
        )

        base_score = metrics_base_val[metric]
        base_val_scores.append(base_score)

        # ==========================
        # MODIFIED MODEL
        # ==========================
        model_mod = model.__class__(**model._kwargs) \
            if hasattr(model, "_kwargs") else model.__class__()

        model_mod.fit(
            df_train_boot[modified_variables + [duration_col, event_col]],
            duration_col=duration_col,
            event_col=event_col
        )

        metrics_mod_val = compute_survival_metrics(
            model_mod,
            df_train_boot[modified_variables + [duration_col, event_col]],
            df_val[modified_variables + [duration_col, event_col]],
            duration_col,
            event_col
        )

        mod_score = metrics_mod_val[metric]
        mod_val_scores.append(mod_score)

        differences.append(mod_score - base_score)

    # Convert to arrays
    base_val_scores = np.array(base_val_scores)
    mod_val_scores = np.array(mod_val_scores)
    differences = np.array(differences)

    # ======================================================
    # Label construction
    # ======================================================
    
    removed_str = ", ".join(variables_to_remove) if variables_to_remove else "None"
    added_str = ", ".join(variables_to_add) if variables_to_add else "None"
    
    base_label = f"Baseline"
    mod_label = (
        f"Modified\n"
        f"Removed: {removed_str}\n"
        f"Added: {added_str}"
    )
    
    # ======================================================
    # PLOT 1: Validation Performance Comparison
    # ======================================================
    
    plt.figure(figsize=(8, 6))
    
    box = plt.boxplot(
        [base_val_scores, mod_val_scores],
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
    for i, scores in enumerate([base_val_scores, mod_val_scores], start=1):
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
        "baseline_val_mean": np.mean(base_val_scores),
        "modified_val_mean": np.mean(mod_val_scores),
        "mean_difference_val": mean_diff,
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "probability_modified_better": prob_better,
        "n_effective_bootstrap": len(differences)
    }

def survival_bootstrap_model_comparison(
    df_train,
    models_dict,
    feature_cols,
    df_val=None,
    duration_col="duration",
    event_col="event",
    start_month_col="start_month",
    n_bootstrap=50
):
    """
    Bootstrap comparison of multiple lifelines survival models.

    Parameters
    ----------
    df_train : pd.DataFrame
    models_dict : dict
        {"ModelName": model_instance}
    df_val : pd.DataFrame or None
        If None → temporal 80/20 split.
    duration_col : str
    event_col : str
    start_month_col : str
    n_bootstrap : int

    Returns
    -------
    dict with bootstrap distributions and ranking summary
    """
    try:
        from lifelines import WeibullAFTFitter, LogNormalAFTFitter, CoxPHFitter
        from lifelines.utils import concordance_index
        
        from sksurv.metrics import (
            concordance_index_ipcw,
            integrated_brier_score,
            brier_score
        )
        from sksurv.util import Surv

    except ImportError as e:
        raise ImportError(
            "Survival analysis features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[survival]\n"
        ) from e

    # --------------------------------------------------
    # Temporal split
    # --------------------------------------------------
    if df_val is None:
        df_train = df_train.sort_values(start_month_col)
        split_index = int(len(df_train) * 0.8)
        df_val = df_train.iloc[split_index:].copy()
        df_train = df_train.iloc[:split_index].copy()

    model_names = list(models_dict.keys())

    ibs_scores = {name: [] for name in model_names}
    cindex_scores = {name: [] for name in model_names}

    # --------------------------------------------------
    # Bootstrap loop
    # --------------------------------------------------
    for b in tqdm(range(n_bootstrap)):

        rng = np.random.default_rng(b)

        df_train_boot = discrete_duration_bootstrap(
            df_train,
            duration_col,
            event_col,
            rng
        )

        if df_train_boot[event_col].sum() < 5:
            continue

        for name, model in models_dict.items():

            model_instance = model.__class__(**model._kwargs) \
                if hasattr(model, "_kwargs") else model.__class__()

            model_instance.fit(
                df_train_boot[feature_cols + [duration_col, event_col]],
                duration_col=duration_col,
                event_col=event_col
            )

            metrics = compute_survival_metrics(
                model_instance,
                df_train_boot,
                df_val,
                duration_col,
                event_col
            )

            ibs_scores[name].append(metrics["ibs"])
            cindex_scores[name].append(metrics["c_index_ipcw"])

    # Convert to arrays
    for name in model_names:
        ibs_scores[name] = np.array(ibs_scores[name])
        cindex_scores[name] = np.array(cindex_scores[name])

    # ======================================================
    # PLOT 1 — IBS
    # ======================================================

    plt.figure(figsize=(9, 6))

    data_ibs = [ibs_scores[name] for name in model_names]

    box = plt.boxplot(data_ibs, patch_artist=True, widths=0.6)

    for patch in box["boxes"]:
        patch.set_alpha(0.6)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    plt.xticks(range(1, len(model_names) + 1), model_names)
    plt.ylabel("IBS")
    plt.title("Integrated Brier Score (Lower is Better)")
    plt.grid(False)
    plt.gcf().set_facecolor("white")
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.show()

    # ======================================================
    # PLOT 2 — IPCW C-index
    # ======================================================

    plt.figure(figsize=(9, 6))

    data_c = [cindex_scores[name] for name in model_names]

    box = plt.boxplot(data_c, patch_artist=True, widths=0.6)

    for patch in box["boxes"]:
        patch.set_alpha(0.6)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    plt.xticks(range(1, len(model_names) + 1), model_names)
    plt.ylabel("C-index IPCW")
    plt.title("IPCW Concordance Index (Higher is Better)")
    plt.grid(False)
    plt.gcf().set_facecolor("white")
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.show()

    # ======================================================
    # Ranking summary
    # ======================================================

    summary = []

    for name in model_names:
        summary.append({
            "Model": name,
            "IBS_mean": ibs_scores[name].mean(),
            "IBS_median": np.median(ibs_scores[name]),
            "Cindex_mean": cindex_scores[name].mean(),
            "Cindex_median": np.median(cindex_scores[name])
        })

    summary_df = pd.DataFrame(summary)

    # Probability of being best
    ibs_matrix = np.vstack([ibs_scores[n] for n in model_names]).T
    c_matrix = np.vstack([cindex_scores[n] for n in model_names]).T

    best_ibs = np.argmin(ibs_matrix, axis=1)
    best_c = np.argmax(c_matrix, axis=1)

    prob_best_ibs = {
        model_names[i]: np.mean(best_ibs == i)
        for i in range(len(model_names))
    }

    prob_best_c = {
        model_names[i]: np.mean(best_c == i)
        for i in range(len(model_names))
    }

    print("\n===== Bootstrap Ranking Summary =====")
    print(summary_df.sort_values("IBS_mean"))

    print("\nProbability of being best (IBS):")
    print(prob_best_ibs)

    print("\nProbability of being best (C-index IPCW):")
    print(prob_best_c)

    return {
        "baseline_vars": base_variables,
        "baseline_val_mean": np.mean(base_val_scores),
        "modified_val_mean": np.mean(mod_val_scores),
        "mean_difference_val": mean_diff,
        "ci_2_5": ci_low,
        "ci_97_5": ci_high,
        "probability_modified_better": prob_better,
        "n_effective_bootstrap": len(differences)
    }