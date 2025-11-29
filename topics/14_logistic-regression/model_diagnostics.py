import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_validity(data):
    """
    Checks for missing values and data integrity.
    
    Assumption 1: Validity
    Data should be relevant and complete.
    """
    print("--- Checking Assumption 1: Validity ---")
    
    if data is None or data.is_empty():
        print("ERROR: Data is empty or None.")
        return
        
    missing_counts = data.null_count()
    total_missing = missing_counts.sum_horizontal().sum()
    
    if total_missing == 0:
        print("✅ No missing values found.")
    else:
        print(f"⚠️ Found {total_missing} missing values.")
        # Show columns with missing values
        for col in missing_counts.columns:
            count = missing_counts[col][0]
            if count > 0:
                print(f"{col}: {count}")
        
    print(f"Data shape: {data.height} rows, {data.width} columns")
    print("-" * 40)

def check_representativeness(data, target_col, model_type='linear'):
    """
    Checks descriptive statistics to assess representativeness.
    
    Assumption 2: Representativeness
    Data should be representative of the population.
    """
    print("--- Checking Assumption 2: Representativeness ---")
    
    if target_col not in data.columns:
        print(f"ERROR: Target column '{target_col}' not found in data.")
        return

    print("Descriptive Statistics for Predictors:")
    print(data.drop(target_col).describe())
    
    print(f"\nTarget Variable '{target_col}' Distribution:")
    if model_type == 'logistic':
        # Check class balance for classification
        counts = data[target_col].value_counts()
        print(counts)
        if counts.height != 2:
            print("⚠️ Target is not binary (expected for Logistic Regression).")
        else:
            # Polars value_counts returns a struct or df with 'count' col usually, 
            # but let's be safe and just get the counts column
            counts_vec = counts["count"] if "count" in counts.columns else counts.select(pl.col(pl.Int64)).to_series()
            ratio = counts_vec.min() / counts_vec.max()
            if ratio < 0.1:
                print("⚠️ Warning: Severe class imbalance detected.")
            else:
                print("✅ Class balance looks reasonable.")
    else:
        # Check range for regression
        desc = data[target_col].describe()
        print(desc)
        # Polars describe output format is different, it's a DF.
        # We can extract min/max.
        min_val = data[target_col].min()
        max_val = data[target_col].max()
        print(f"Range: {min_val} to {max_val}")
        
    print("-" * 40)

def check_linearity(data, features, target_col, model_type='linear'):
    """
    Checks for linearity between predictors and the outcome (or log-odds).
    
    Assumption 3: Linearity
    - Linear Regression: Y should be linearly related to X.
    - Logistic Regression: Log-odds of Y should be linearly related to X.
    """
    print("--- Checking Assumption 3: Linearity ---")
    
    num_features = len(features)
    if num_features == 0:
        print("No features provided.")
        return

    # Set up the plot grid
    cols = 2
    rows = (num_features + 1) // cols
    if rows == 0: rows = 1
    if num_features == 1: cols = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if num_features > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Convert only necessary columns to pandas for plotting
        plot_data = data.select([feature, target_col]).to_pandas()
        
        if model_type == 'linear':
            # Scatter plot with regression line
            sns.regplot(x=feature, y=target_col, data=plot_data, ax=ax, 
                        scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            ax.set_title(f"Linearity: {feature} vs {target_col}")
            
        elif model_type == 'logistic':
            # 1. Scatter with Lowess (Probability scale)
            sns.regplot(x=feature, y=target_col, data=plot_data, ax=ax, 
                        logistic=True, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            ax.set_title(f"Logistic Link: {feature} vs {target_col}")
            
    # Hide unused subplots
    if num_features > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()
    print("✅ Visual inspection required. \n- Linear: Look for points scattered around the straight line.\n- Logistic: Look for an S-shaped curve fitting the data.")
    print("-" * 40)

def check_independence(data, target_col, time_variable=None):
    """
    Checks for independence of observations.
    
    Assumption 4: Independence
    Observations should be independent. Look for patterns in residuals (or Y) over time/index.
    """
    print("--- Checking Assumption 4: Independence ---")
    
    plt.figure(figsize=(10, 5))
    
    if time_variable and time_variable in data.columns:
        # Plot against time variable
        plot_data = data.select([time_variable, target_col]).to_pandas()
        sns.scatterplot(x=time_variable, y=target_col, data=plot_data)
        plt.title(f"Independence Check: {target_col} vs {time_variable}")
        plt.xlabel(time_variable)
    else:
        # Plot against index
        # Polars doesn't have a default index, use row number
        y_values = data[target_col].to_numpy()
        plt.plot(range(len(y_values)), y_values, marker='o', linestyle='none', alpha=0.6)
        plt.title(f"Independence Check: {target_col} vs Index")
        plt.xlabel("Index (Row Number)")
        
    plt.ylabel(target_col)
    plt.show()
    
    print("✅ Visual inspection required. Look for patterns (e.g., trends, cycles) that suggest dependence.")
    print("-" * 40)

def check_constant_variance(data, features, target_col, model_type='linear'):
    """
    Checks for homoscedasticity (constant variance).
    
    Assumption 5: Constant Variance
    - Linear: Variance of errors should be constant.
    - Logistic: Not applicable (variance depends on mean).
    """
    print("--- Checking Assumption 5: Constant Variance ---")
    
    if model_type == 'logistic':
        print("ℹ️ Constant variance is not an assumption for Logistic Regression (variance depends on the mean).")
        print("-" * 40)
        return

    # For linear regression, we want to see if the spread of Y changes with X.
    
    num_features = len(features)
    cols = 2
    rows = (num_features + 1) // cols
    if rows == 0: rows = 1
    if num_features == 1: cols = 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if num_features > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    for i, feature in enumerate(features):
        ax = axes[i]
        plot_data = data.select([feature, target_col]).to_pandas()
        sns.scatterplot(x=feature, y=target_col, data=plot_data, ax=ax, alpha=0.5)
        ax.set_title(f"Spread: {feature} vs {target_col}")
        
    if num_features > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
    plt.tight_layout()
    plt.show()
    
    print("✅ Visual inspection required. Look for 'fanning' or 'funnel' shapes (heteroscedasticity).")
    print("-" * 40)

def check_normality(data, target_col, model_type='linear'):
    """
    Checks for normality of errors (approximated by checking Y).
    
    Assumption 6: Normality
    - Linear: Errors should be normally distributed. (Checking Y is a proxy).
    - Logistic: Not applicable.
    """
    print("--- Checking Assumption 6: Normality ---")
    
    if model_type == 'logistic':
        print("ℹ️ Normality of errors is not an assumption for Logistic Regression.")
        print("-" * 40)
        return

    # Check distribution of Target Variable
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    y_values = data[target_col].to_numpy()
    
    # Histogram
    sns.histplot(y_values, kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {target_col}")
    
    # Q-Q Plot
    stats.probplot(y_values, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of {target_col}")
    
    plt.tight_layout()
    plt.show()
    
    print("⚠️ Note: The strict assumption is about *errors*, not the raw outcome.")
    print("However, if Y is roughly normal, linear regression is often appropriate.")
    print("✅ Visual inspection required. Look for a bell curve and points on the line in Q-Q plot.")
    print("-" * 40)

def check_identifiability(data, features):
    """
    Checks for multicollinearity using VIF.
    
    Assumption 7: Identifiability
    Predictors should not be perfectly correlated (no multicollinearity).
    """
    print("--- Checking Assumption 7: Identifiability (Multicollinearity) ---")
    
    if len(features) < 2:
        print("ℹ️ Only one predictor. Multicollinearity is not possible.")
        print("-" * 40)
        return
        
    # Select only numeric features for VIF
    # Polars selector for numeric types
    import polars.selectors as cs
    numeric_data = data.select(features).select(cs.numeric())
    
    if numeric_data.width < len(features):
        print("⚠️ Warning: Non-numeric features detected. VIF calculation skipped for non-numeric columns.")
        
    if numeric_data.is_empty():
        print("ERROR: No numeric features for VIF calculation.")
        return
        
    # Drop rows with missing values for VIF calculation
    numeric_data = numeric_data.drop_nulls()
    
    # Convert to pandas/numpy for VIF
    df_pandas = numeric_data.to_pandas()
    
    import pandas as pd # Needed for the VIF dataframe construction
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_pandas.columns
    vif_data["VIF"] = [variance_inflation_factor(df_pandas.values, i) 
                       for i in range(df_pandas.shape[1])]
    
    print(vif_data)
    
    high_vif = vif_data[vif_data["VIF"] > 5]
    if not high_vif.empty:
        print("\n⚠️ High Multicollinearity Detected (VIF > 5):")
        print(high_vif)
    else:
        print("\n✅ No severe multicollinearity detected (all VIF <= 5).")
        
    print("-" * 40)

