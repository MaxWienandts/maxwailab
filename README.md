## Bootstrap ML diagnostics + statistical inference + Spark

A lightweight toolkit for statistically robust model diagnostics using bootstrap resampling, with both in-memory and distributed (PySpark) support.

The library provides utilities for:

- supervised tree binning  
- bootstrap-based feature selection  
- model stability analysis  
- hyperparameter sensitivity analysis  
- statistical feature diagnostics (e.g., logistic relevance, missing analysis)  
- scalable data diagnostics with PySpark  

The toolkit focuses on reducing overfitting and improving model interpretability by leveraging bootstrap distributions and statistical inference rather than single-point estimates.

---

# Installation

| Option | Command |
|--------|---------|
| Core (minimal dependencies) | `pip install maxwailab` |
| Survival Module Only | `pip install maxwailab[survival]` |
| PySpark Module Only | `pip install maxwailab[pyspark]` |
| Everything (core + all optional) | `pip install maxwailab[all]` |
| Core from GitHub | `pip install git+https://github.com/MaxWienandts/maxwailab.git` |
| GitHub with survival extras | `pip install "git+https://github.com/MaxWienandts/maxwailab.git#egg=maxwailab[survival]"` |
| GitHub with PySpark extras | `pip install "git+https://github.com/MaxWienandts/maxwailab.git#egg=maxwailab[pyspark]"` |

---

# Core Philosophy

Most ML workflows rely on **single train/validation splits**.

This library instead uses **bootstrap resampling** to estimate:

* performance **distributions**
* feature **selection stability**
* hyperparameter **robustness**

Benefits:

* reduces variance from a single split
* identifies unstable variables
* provides confidence intervals for model performance

---

# Workflow Overview

Typical modeling workflow using this library:

```
1️⃣ Supervised binning (optional)

tree_supervised_binning
bootstrap_tree_binning_auc_analysis
plot_target_mean_by_binned_variable


2️⃣ Feature selection

bootstrap_lightgbm_forward_selection
bootstrap_model_variable_comparison_paired
bootstrap_survival_forward_selection
survival_bootstrap_model_comparison
    
3️⃣ Diagnostics

performance_forward_selection_boxplot
variable_frequency_forward_selection


4️⃣ Extract best variables

top_k_forward_selection_variables_by_frequency_usage
top_k_variables_by_forward_selection_boxplot


5️⃣ Hyperparameter analysis

lightgbm_hyperparameter_auc_curve_bootstrap


6️⃣ PySpark Data Diagnostics
pyspark_missing_values_table
pyspark_minmax_value
pyspark_compare_columns
pyspark_value_counts_spark
pyspark_missing_by_group
pyspark_logistic_feature_significance
```

---

# LightGBM Classification Example

```python
import maxwailab

# Forward selection with bootstrap
hyperparameters = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "is_unbalanced": True
}

result_bootstrap = maxwailab.bootstrap_lightgbm_forward_selection(
    df=df,
    target="target",
    n_bootstrap=10,
    n_max_variables=10,
    metric_to_optimize="auc_roc",
    hyperparameters=hyperparameters
)

# Analyze performance stability
maxwailab.performance_forward_selection_boxplot(result_bootstrap["auc_roc"], "AUC")

# Variable selection stability
maxwailab.variable_frequency_forward_selection(
    result_bootstrap["variables"],
    n_bootstraps=10,
    figsize = (15, 15)
)

# Extract best variables
top_vars = maxwailab.top_k_forward_selection_variables_by_frequency_usage(
    result_bootstrap["variables"],
    n_bootstraps=10,
    k=8
)
print(top_vars)
# Or
vars_best, perf = maxwailab.top_k_variables_by_forward_selection_boxplot(
    result_bootstrap,
    k=2,
    metric="auc_roc"
)
print(vars_best)
print()
print(perf)
```

## Paired Bootstrap Comparison (LightGBM)

Compare two models: baseline vs modified (adding/removing variables):
```python
maxwailab.bootstrap_model_variable_comparison_paired_lgbm(
    df_train,
    base_variables=['worst perimeter', 'worst smoothness', 'worst texture', 'mean texture', 'mean concave points', 'worst radius', 'radius error', 'worst concavity'],
    target_col='target',
    df_val=df_val,
    start_month_col=None,
    variables_to_remove=['worst smoothness', 'worst texture'],
    variables_to_add=['mean compactness', 'mean area'],
    n_bootstrap=100,
    metric="auc",
    hyperparameters=None
)
```
Generates:
- Validation performance distributions
- Paired bootstrap difference distribution
- Statistical summary (mean, 95% CI, probability of improvement)

## Tree-based Supervised Binning
```python
from maxwailab import tree_supervised_binning

best_bin_mean_radius_bin_size = tree_supervised_binning(
    df = df_train,
    feature='mean radius',
    target='target',
    max_leaf_nodes=5
)
bins = best_bin_mean_radius_bin_size['thresholds']
print(
    f"mean radius: "
    f"{bins}"
)
print()
display(best_bin_mean_radius_bin_size['bin_summary'])

# Bootstrap binning stability
bootstrap_tree_binning_auc_analysis_mean_radius = maxwailab.bootstrap_tree_binning_auc_analysis(
    df_train,
    df_val,
    feature = 'mean radius',
    target = 'target',
    max_leaf_nodes_max = 7,
    n_bootstrap = 30,
    min_samples_leaf = 0.10,
    random_state = 42,
    plot = True,
)

# Hyperparameter Sensitivity Analysis
analysis = maxwailab.lightgbm_hyperparameter_auc_curve_bootstrap(
    X_train,
    y_train,
    X_val,
    y_val,
    hyperparameters=hyperparameters,
    hyperparameter_name="num_leaves",
    hyperparameter_values=[5,10,20,40,80],
    n_bootstrap=200
)

print(analysis["results"])
print()
print(analysis["best_hyperparameters"])
```

## Analyze target behavior across variable ranges
```python
# Define bins (no need for -inf / +inf)
bins = [0, 18, 30, 50, 80]
maxwailab.plot_target_mean_by_binned_variable(
    df=df_train,
    target="target",
    variable="mean radius",
    bins=bins
)
```

- Visualizes target mean per bin
- Displays observation count and percentage
- Useful for feature understanding and pre-binning analysis

## Survival Analysis Workflows
Bootstrap Forward Selection for Survival Models
```python
hyperparameters = {
    "penalizer": 0.01
}
results_bootstrap_survival_forward_selection = maxwailab.bootstrap_survival_forward_selection(
    df_train=df,
    duration_col="duration",
    event_col="event",
    start_month_col="start_year_month",
    model=WeibullAFTFitter(penalizer=0.01),  #cox_breslow, cox_spline, aft_lognormal, aft_weibull
    n_bootstrap=30,
    n_max_variables=7,
    metric_to_optimize="ibs",
    df_val=None  # triggers temporal split
)

# Analyze performance stability
maxwailab.performance_forward_selection_boxplot(results_bootstrap_survival_forward_selection["ibs"], "Integrated Brier Score")

# Variable selection stability
maxwailab.variable_frequency_forward_selection(
    results_bootstrap_survival_forward_selection["variables"],
    n_bootstraps=30,
    figsize = (7, 7)
)

# Extract best variables
top_vars = maxwailab.top_k_forward_selection_variables_by_frequency_usage(result_survival["variables"], n_bootstraps=30, k=10)
# Or
top_vars = maxwailab.top_k_variables_by_forward_selection_boxplot(result_survival["variables"], n_bootstraps=30, k=10)
```

## Paired Bootstrap Comparison for Survival Models
```python
comparison_surv = maxwailab.bootstrap_model_variable_comparison_paired(
    df_train = df,
    df_val = None,
    duration_col = 'duration',
    event_col = 'event',
    start_month_col = 'start_year_month',
    model=WeibullAFTFitter(penalizer = 0.01),  #cox_breslow, cox_spline, aft_lognormal, aft_weibull,
    base_variables=['fin', 'age', 'race', 'wexp'],
    variables_to_remove=['race', 'age'],
    variables_to_add=['mar', 'paro'],
    n_bootstrap=50,
    metric="ibs",
)
```

Generates:
- Baseline vs Modified model performance distribution
- Paired difference plot
- Statistical inference summary

## Compare Multiple Survival Models
```python
models = {
    "Cox": CoxPHFitter(penalizer=0.01),
    "Weibull AFT": WeibullAFTFitter(penalizer=0.01),
    "LogNormal AFT": LogNormalAFTFitter(penalizer=0.01),
}

results = maxwailab.survival_bootstrap_model_comparison(
    df_train=df,
    feature_cols=['fin', 'age', 'race', 'wexp'],
    models_dict=models,
    df_val=None,
    duration_col="duration",
    event_col="event",
    start_month_col="start_year_month",
    n_bootstrap=50
)
```
Outputs:
- Bootstrap distributions per model
- Ranking summary

---

### 🔍 Additional Utilities

The examples above cover the core functionality of the library.  
`maxwailab` also includes several additional utilities for:

- PySpark-based data diagnostics  
- statistical feature analysis  
- extended bootstrap evaluations  
- survival modeling workflows  

For a complete list of available functions and usage examples, refer to the `notebooks/` directory in the repository, which contains practical, end-to-end implementations.

---

# Module Structure
```
maxwailab
│
├── binning
│   ├── tree_supervised_binning
│   ├── bootstrap_tree_binning_auc_analysis
│   ├── plot_target_mean_by_binned_variable
│   ├── pandas_one_hot_encode,
│   └── pandas_round_number_strings,
│
├── feature_selection
│   ├── bootstrap_lightgbm_forward_selection
│   ├── performance_forward_selection_boxplot
│   ├── variable_frequency_forward_selection
│   ├── top_k_forward_selection_variables_by_frequency_usage
│   ├── top_k_variables_by_forward_selection_boxplot
│   └── bootstrap_model_variable_comparison_paired_lgbm
│
├── hyperparameter_analysis
│   └── lightgbm_hyperparameter_auc_curve_bootstrap
│
├── survival_feature_selection
│   ├── bootstrap_survival_forward_selection
│   ├── bootstrap_model_variable_comparison_paired
│   └── survival_bootstrap_model_comparison
│
├── pyspark_basic_functions
│   ├── pyspark_missing_values_table
│   ├── pyspark_minmax_value
│   ├── pyspark_compare_columns
│   ├── pyspark_value_counts_spark
│   ├── pyspark_missing_by_group
│   ├── pyspark_logistic_feature_significance
│   ├── pyspark_one_hot_encode,
│   ├── pyspark_print_shape,
│   └── pyspark_round_number_strings,

```
        
---

# When to Use This Library

This library is particularly useful for:

* **credit risk models**
* **tabular ML problems**
* **high-stakes predictive modeling**
* **interpretable ML workflows**
* **Using large-scale datasets with PySpark**
---

# License

MIT License