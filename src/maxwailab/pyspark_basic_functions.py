import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from __future__ import annotations

try:
    from pyspark.sql import DataFrame, Window
    import pyspark.sql.functions as F
    from pyspark.sql import DataFrame
    from pyspark.sql.functions import col, count, when, isnan
    from pyspark.sql.types import NumericType
    import statsmodels.api as sm
except ImportError:
    # pyspark dependencies not installed
    pass



def pyspark_print_shape(
    df: DataFrame,
    show_result: bool = False
) -> dict:
    """
    Get the shape of a PySpark DataFrame.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    show_result : bool, default=False
        If True, prints the number of rows and columns.

    Returns
    -------
    dict
        {
            "n_rows": int,
            "n_columns": int
        }

    Notes
    -----
    - Triggers a Spark action (count), which may be expensive for large datasets.
    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
        
    
    
    # -------------------------------
    # Compute shape
    # -------------------------------
    n_rows = df.count()
    n_columns = len(df.columns)

    result = {
        "n_rows": n_rows,
        "n_columns": n_columns
    }

    # -------------------------------
    # Optional display
    # -------------------------------
    if show_result:
        print(f"Number of rows   : {n_rows:,}")
        print(f"Number of columns: {n_columns:,}")

    return result

# --------------------------------------------------------------------------------------------

def pyspark_missing_values_table(df: DataFrame) -> pd.DataFrame:
    """
    Generate a table with the count and percentage of missing values per column.

    Missing values are defined as:
    - NULL values for all column types
    - NaN values for numeric columns

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input Spark DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with:
        - Column: column name
        - Missing Values: count of missing values
        - Missing Values %: percentage of missing values

    Notes
    -----
    - The function converts results to Pandas for easier manipulation.
    - Suitable for exploratory data analysis (EDA).
    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
    
    # Build expressions to count missing values per column
    mis_val_exprs = []
    dtype_map = dict(df.dtypes)

    for c in df.columns:
        col_type = dtype_map[c]

        # For numeric columns, check both NULL and NaN
        if col_type in ['double', 'float', 'int', 'bigint', 'smallint', 'tinyint', 'decimal']:
            expr = count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
        else:
            # For non-numeric columns, check only NULL
            expr = count(when(col(c).isNull(), c)).alias(c)

        mis_val_exprs.append(expr)

    # Compute missing values count
    mis_val = df.select(mis_val_exprs)

    # Compute percentage of missing values
    total_count = df.count()
    mis_val_percent = mis_val.select([
        (col(c) / total_count * 100).alias(c) for c in mis_val.columns
    ])

    # Convert to Pandas for easier formatting
    mis_val_pd = mis_val.toPandas().transpose().reset_index()
    mis_val_pd.columns = ['Column', 'Missing Values']

    mis_val_percent_pd = mis_val_percent.toPandas().transpose().reset_index()
    mis_val_percent_pd.columns = ['Column', 'Missing Values %']
    mis_val_percent_pd['Missing Values %'] = mis_val_percent_pd['Missing Values %'].round(2)

    # Merge results
    result = mis_val_pd.merge(mis_val_percent_pd, on='Column')

    # Sort by percentage of missing values
    result.sort_values(by='Missing Values %', ascending=False, inplace=True)

    return result

# --------------------------------------------------------------------------------------------

def pyspark_minmax_value(df: DataFrame, col: str):
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql.functions import count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e


    df = df.filter(~(F.col(col).isNull() | isnan(F.col(col))))
    max_value = df.agg(F.max(col)).collect()[0][0]
    min_value = df.agg(F.min(col)).collect()[0][0]

    # Display the maximum value
    print(f"{col} min value: {min_value}")
    print(f"{col} max value: {max_value}")
    return {"min_value":min_value, "max_value":max_value}

# --------------------------------------------------------------------------------------------

def pyspark_compare_columns(
    df: DataFrame,
    col1: str,
    col2: str,
    show_results: bool = True
) -> dict:
    """
    Compare two columns in a PySpark DataFrame.

    Returns:
    -------
    dict with:
        - value_comparison: comparison of non-missing values
        - missing_comparison: comparison of NULL/NaN alignment
    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
    
    # Helper: detect missing values (NULL or NaN)
    def is_missing(df: DataFrame, col_name: str):
        dtype = dict(df.dtypes)[col_name]
    
        if dtype in ['double', 'float', 'int', 'bigint', 'smallint', 'tinyint', 'decimal']:
            return col(col_name).isNull() | isnan(col(col_name))
        else:
            return col(col_name).isNull()

    # -------------------------------
    # 1. VALUE COMPARISON (ONLY VALID DATA)
    # -------------------------------
    missing_col1 = is_missing(df, col1)
    missing_col2 = is_missing(df, col2)
    
    df_valid = df.filter(~(missing_col1 | missing_col2))

    df_valid = df_valid.withColumn(
        "comparison",
        when(col(col1) == col(col2), "equal").otherwise("different")
    )

    value_result = df_valid.groupBy("comparison").agg(
        count("*").alias("count")
    )

    total_valid = df_valid.count()

    if total_valid > 0:
        value_result = value_result.withColumn(
            "percentage", (col("count") / total_valid) * 100
        )

    # -------------------------------
    # 2. MISSING ALIGNMENT COMPARISON
    # -------------------------------
    df_missing = df.withColumn(
        "missing_comparison",
        when(is_missing(df, col1) & is_missing(df, col2), "both_missing")
        .when(is_missing(df, col1) & ~is_missing(df, col2), f"{col1}_missing_only")
        .when(~is_missing(df, col1) & is_missing(df, col2), f"{col2}_missing_only")
        .otherwise("none_missing")
    )

    missing_result = df_missing.groupBy("missing_comparison").agg(
        count("*").alias("count")
    )

    total_rows = df.count()

    missing_result = missing_result.withColumn(
        "percentage", (col("count") / total_rows) * 100
    )

    if show_results == True:
        value_result.show()
        missing_result.show()
    return {
        "value_comparison": value_result,
        "missing_comparison": missing_result
    }

# --------------------------------------------------------------------------------------------

def pyspark_value_counts_spark(
    df: DataFrame,
    column: str,
    show_results: bool = True,
    include_nulls: bool = True
) -> DataFrame:
    """
    Compute count and percentage distribution of values in a column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    column : str
        Column to analyze.
    show_results : bool, default=False
        If True, displays the result using .show().
    include_nulls : bool, default=True
        Whether to include NULL values in the analysis.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with:
        - column
        - count
        - percentage

    Notes
    -----
    - Uses a single pass over data (no df.count()).
    - Suitable for large-scale datasets.
    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
    
    # Optionally filter NULL values
    if not include_nulls:
        df = df.filter(col(column).isNotNull())

    # Count occurrences
    total_count = df.count()

    result = df.groupBy(column).agg(
        count("*").alias("count")
    )

    result = result.withColumn(
        "percentage",
        (col("count") / F.lit(total_count)) * 100
    )


    # Sort by count descending
    result = result.orderBy(col("count").desc())

    # Optional display
    if show_results:
        result.show(truncate=False)

    return result

# --------------------------------------------------------------------------------------------

def pyspark_missing_by_group(
    df: DataFrame,
    group_col: str,
    target_col: str,
    show_results: bool = True
) -> DataFrame:
    """
    Compute missing value statistics of a column grouped by another column.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    group_col : str
        Column used for grouping (e.g., 'YearMonth').
    target_col : str
        Column for which missing values are computed.
    show_results : bool, default=False
        If True, displays the result using .show().

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with:
        - group_col
        - total_count
        - missing_count
        - missing_percentage
    """

    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql import DataFrame
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e

    # Detect if column is numeric (to safely apply isnan)
    field = df.schema[target_col]
    is_numeric = isinstance(field.dataType, NumericType)

    # Missing condition (type-safe)
    if is_numeric:
        missing_expr = col(target_col).isNull() | isnan(col(target_col))
    else:
        missing_expr = col(target_col).isNull()

    # Single aggregation pass (optimized)
    result = df.groupBy(group_col).agg(
        count("*").alias("total_count"),
        F.sum(when(missing_expr, 1).otherwise(0)).alias("missing_count")
    )

    # Compute percentage
    result = result.withColumn(
        "missing_percentage",
        F.round((col("missing_count") / col("total_count")) * 100, 2)
    )

    # Order by group column
    result = result.orderBy(col(group_col))

    # Optional display
    if show_results:
        result.show(truncate=False)

    return result

# --------------------------------------------------------------------------------------------

def pyspark_one_hot_encode(
    df: DataFrame,
    categorical_cols: list,
    drop_strategy: str = "least_frequent",
    drop_if_contains: str = "inform",
    show_removed_categories: bool = False
) -> DataFrame:
    """
    Perform one-hot encoding on categorical columns with automatic baseline removal.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    categorical_cols : list
        List of categorical column names.
    drop_strategy : str, default="least_frequent"
        Strategy to select category to drop:
        - "least_frequent": drop category with lowest count
        - "none": do not drop any category
    drop_if_contains : str, default="inform"
        If a category contains this string (case-insensitive),
        it will be prioritized for removal.
    show_removed_categories : bool, default=False
        If True, prints which category was dropped per column.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with one-hot encoded columns added.

    Notes
    -----
    - Drops one category to prevent multicollinearity (dummy variable trap).
    - Sanitizes column names to be Spark-safe.
    """

    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql import DataFrame
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
        
    def sanitize_col_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "", name)

    for col_name in categorical_cols:

        # -------------------------------
        # 1. Get category counts (small collect)
        # -------------------------------
        counts_df = (
            df.groupBy(col_name)
            .agg(F.count("*").alias("count"))
        )

        counts = counts_df.collect()  # still needed, but now minimal

        if len(counts) <= 1:
            continue

        categories = [row[col_name] for row in counts]

        # -------------------------------
        # 2. Determine category to drop
        # -------------------------------
        drop_cat = None

        if drop_strategy != "none":
            # Priority: categories containing keyword
            if drop_if_contains:
                inform_cats = [
                    cat for cat in categories
                    if cat and drop_if_contains.lower() in str(cat).lower()
                ]
                if inform_cats:
                    drop_cat = inform_cats[0]

            # Otherwise: least frequent
            if drop_cat is None and drop_strategy == "least_frequent":
                drop_cat = min(counts, key=lambda x: x["count"])[col_name]

        if show_removed_categories:
            print(f"[{col_name}] dropped category: {drop_cat}")

        # -------------------------------
        # 3. Create dummy columns
        # -------------------------------
        for cat in categories:
            if cat == drop_cat:
                continue

            safe_cat = sanitize_col_name(str(cat))
            dummy_col = f"{col_name}_{safe_cat}"

            df = df.withColumn(
                dummy_col,
                F.when(F.col(col_name) == cat, F.lit(1)).otherwise(F.lit(0))
            )

    return df

# --------------------------------------------------------------------------------------------

def pyspark_round_number_strings(
    df: DataFrame,
    columns: list = None,
    decimals: int = 2,
) -> DataFrame:
    """
    Round numeric values inside string interval columns.

    Example:
    --------
    "1.2345" → "1.23"
    "(1.2345, 5.6789]" → "(1.23, 5.68]"

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to process. If None, automatically selects string columns
        with low cardinality.
    decimals : int, default=2
        Number of decimal places for rounding.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame with transformed columns.

    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql import DataFrame
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType, StringType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e

    # -------------------------------
    # Regex pattern for floats
    # -------------------------------
    pattern = r'-?\d+\.\d+'

    # -------------------------------
    # 4. Efficient UDF (vectorized alternative not available)
    # -------------------------------
    def round_numbers_in_string(s: str):
        if s is None:
            return None

        def repl(m):
            return f"{float(m.group(0)):.{decimals}f}"

        return re.sub(pattern, repl, s)

    round_udf = F.udf(round_numbers_in_string, StringType())

    for col_name in columns:
        df = df.withColumn(col_name, round_udf(F.col(col_name)))

    return df
    
# --------------------------------------------------------------------------------------------

def pyspark_logistic_feature_significance(
    df: DataFrame,
    explanatory_var: str,
    response_var: str,
    show_results: bool = True,
    max_rows: int = 100000,
    standardize: bool = True
) -> dict:
    """
    Evaluate feature significance using logistic regression.

    Optionally standardizes the explanatory variable to allow
    coefficient comparability across variables.
    """
    try:
        from pyspark.sql import DataFrame, Window
        import pyspark.sql.functions as F
        from pyspark.sql import DataFrame
        from pyspark.sql.functions import col, count, when, isnan
        from pyspark.sql.types import NumericType
        import statsmodels.api as sm

    except ImportError as e:
        raise ImportError(
            "PySpark features require optional dependencies. "
            "Install them with:\n\n"
            "    pip install maxwailab[pyspark]\n"
        ) from e
        
    # -------------------------------
    # 1. Select columns
    # -------------------------------
    df_sel = df.select(explanatory_var, response_var)

    # -------------------------------
    # 2. Missing handling
    # -------------------------------
    schema = df.schema

    def is_missing(col_name):
        field = schema[col_name]
        if isinstance(field.dataType, NumericType):
            return F.col(col_name).isNull() | F.isnan(F.col(col_name))
        else:
            return F.col(col_name).isNull()

    df_sel = df_sel.filter(
        ~(is_missing(explanatory_var) | is_missing(response_var))
    )

    # -------------------------------
    # 3. Limit size
    # -------------------------------
    df_sel = df_sel.limit(max_rows)

    # -------------------------------
    # 4. Convert to Pandas
    # -------------------------------
    pdf = df_sel.toPandas()

    if pdf.empty:
        raise ValueError("No valid data after filtering missing values.")

    # -------------------------------
    # 5. Prepare data
    # -------------------------------
    y = pdf[response_var].astype(float)
    X = pdf[[explanatory_var]].astype(float)

    # Validate binary target
    unique_vals = set(y.unique())
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError(f"{response_var} must be binary (0/1). Found: {unique_vals}")

    # -------------------------------
    # 6. Standardization (KEY STEP)
    # -------------------------------
    if standardize:
        std = X[explanatory_var].std()

        if std == 0:
            raise ValueError(f"{explanatory_var} has zero variance.")

        X[explanatory_var] = (
            X[explanatory_var] - X[explanatory_var].mean()
        ) / std

    # Add intercept
    X = sm.add_constant(X)

    # -------------------------------
    # 7. Fit model
    # -------------------------------
    model = sm.Logit(y, X)

    try:
        result = model.fit(disp=False)
    except Exception as e:
        raise RuntimeError(f"Model failed to converge: {e}")

    # -------------------------------
    # 8. Extract results
    # -------------------------------
    coef = float(result.params[explanatory_var])
    p_value = float(result.pvalues[explanatory_var])

    output = {
        "coefficient": coef,
        "p_value": p_value,
        "n_obs": int(len(pdf)),
        "standardized": standardize
    }

    # -------------------------------
    # 9. Optional display
    # -------------------------------
    if show_results:
        print("\nLogistic Regression Summary")
        print(result.summary())

    return output

# --------------------------------------------------------------------------------------------
