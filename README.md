# Bootstrap ML Diagnostics

A lightweight toolkit for **statistically robust model diagnostics** using **bootstrap resampling**, with utilities for:

* supervised tree binning
* bootstrap-based feature selection
* model stability analysis
* hyperparameter sensitivity analysis

The library focuses on **reducing overfitting and improving model interpretability** through **bootstrap distributions rather than single-point estimates**.

---

# Installation

| Option | Command |
|--------|---------|
| Core (minimal dependencies) | `pip install maxwailab` |
| Survival Module Only | `pip install maxwailab[survival]` |
| Everything (core + all optional) | `pip install maxwailab[all]` |
| Core from GitHub | `pip install git+https://github.com/MaxWienandts/maxwailab.git` |
| GitHub with survival extras | `pip install "git+https://github.com/MaxWienandts/maxwailab.git#egg=maxwailab[survival]"` |

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


2️⃣ Feature selection

bootstrap_lightgbm_forward_selection


3️⃣ Diagnostics

performance_forward_selection_boxplot
variable_frequency_forward_selection


4️⃣ Extract best variables

top_k_forward_selection_variables
top_k_variables_by_forward_selection_boxplot


5️⃣ Hyperparameter analysis

lightgbm_hyperparameter_auc_curve_bootstrap
```

---

# LightGBM Classification Example

```python
import maxwailab

# Forward selection with bootstrap
result_bootstrap = maxwailab.bootstrap_lightgbm_forward_selection(
    df=data,
    target="target",
    n_bootstrap=30,
    n_max_variables=15,
    metric_to_optimize="auc_roc",
    hyperparameters=lgb_params
)

# Analyze performance stability
maxwailab.performance_forward_selection_boxplot(result_bootstrap["auc_roc"], "AUC")

# Variable selection stability
maxwailab.variable_frequency_forward_selection(result_bootstrap["variables"], n_bootstraps=30)

# Extract best variables
top_vars = maxwailab.top_k_forward_selection_variables_by_frequency_usage(result_bootstrap["variables"], n_bootstraps=30, k=10)
# Or
top_vars = maxwailab.top_k_variables_by_forward_selection_boxplot(result_bootstrap["variables"], n_bootstraps=30, k=10)
```

## Paired Bootstrap Comparison (LightGBM)

Compare two models: baseline vs modified (adding/removing variables):
```python
comparison = maxwailab.bootstrap_model_variable_comparison_paired_lgbm(
    df_train=df_train,
    base_variables=["var1", "var2"],
    variables_to_add=["var3"],
    variables_to_remove=["var2"],
    target_col="target",
    n_bootstrap=100,
    metric="auc",
    hyperparameters=lgb_params
)
```
Generates:
- Validation performance distributions
- Paired bootstrap difference distribution
- Statistical summary (mean, 95% CI, probability of improvement)

## Tree-based Supervised Binning
```python
from maxwailab import tree_supervised_binning

tree_supervised_binning(df=data, feature="age", target="target", max_leaf_nodes=5)

# Bootstrap binning stability
bootstrap_tree_binning_auc_analysis(df_train, df_val, feature="age", target="target")
Hyperparameter Sensitivity Analysis
lightgbm_hyperparameter_auc_curve_bootstrap(
    X_train, y_train, X_val, y_val,
    hyperparameters=lgb_params,
    hyperparameter_name="num_leaves",
    hyperparameter_values=[5,10,20,40],
    n_bootstrap=50
)
```

## Survival Analysis Workflows
Bootstrap Forward Selection for Survival Models
```python
result_survival = maxwailab.bootstrap_survival_forward_selection(
    df_train=df_train,
    duration_col="duration",
    event_col="event",
    start_month_col="start_month",
    model_type="cox_breslow",
    n_bootstrap=50,
    n_max_variables=10,
    metric_to_optimize="c_index",
    hyperparameters=cox_params
)

# Analyze performance stability
maxwailab.performance_forward_selection_boxplot(result_survival["auc_roc"], "AUC")

# Variable selection stability
maxwailab.variable_frequency_forward_selection(result_survival["variables"], n_bootstraps=30)

# Extract best variables
top_vars = maxwailab.top_k_forward_selection_variables_by_frequency_usage(result_survival["variables"], n_bootstraps=30, k=10)
# Or
top_vars = maxwailab.top_k_variables_by_forward_selection_boxplot(result_survival["variables"], n_bootstraps=30, k=10)
```

## Paired Bootstrap Comparison for Survival Models
```python
comparison_surv = maxwailab.bootstrap_model_variable_comparison_paired(
    df_train=df_train,
    model_type="cox_breslow",
    base_variables=["var1", "var2"],
    variables_to_add=["var3"],
    variables_to_remove=["var2"],
    n_bootstrap=50,
    metric="c_index"
)
```

Generates:
- Baseline vs Modified model performance distribution
- Paired difference plot
- Statistical inference summary

## Compare Multiple Survival Models
```python
models_dict = {
    "Cox": CoxModel(),
    "AFT": AFTModel()
}

comparison_multi = maxwailab.survival_bootstrap_model_comparison(
    df_train=df_train,
    models_dict=models_dict,
    feature_cols=["var1", "var2", "var3"],
    n_bootstrap=50
)
```
Outputs:
- Bootstrap distributions per model
- Ranking summary

# Module Structure

```
maxwailab
│
├── binning
│   ├── tree_supervised_binning
│   └── bootstrap_tree_binning_auc_analysis
│
├── feature_selection
│   ├── bootstrap_lightgbm_forward_selection
│   ├── performance_forward_selection_boxplot
│   ├── variable_frequency_forward_selection
│   ├── top_k_forward_selection_variables_by_frequency_usage
│   └── top_k_variables_by_forward_selection_boxplot
│   └── bootstrap_model_variable_comparison_paired_lgbm
│
└── hyperparameter_analysis
    └── lightgbm_hyperparameter_auc_curve_bootstrap
│
└── survival_feature_selection
    └── bootstrap_survival_forward_selection
    └── bootstrap_model_variable_comparison_paired
    └── survival_bootstrap_model_comparison
```
   
---

# When to Use This Library

This library is particularly useful for:

* **credit risk models**
* **tabular ML problems**
* **high-stakes predictive modeling**
* **interpretable ML workflows**

---

# License

MIT License