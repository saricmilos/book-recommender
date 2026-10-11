# Core libraries
import math

# Data manipulation
import pandas as pd        # DataFrames for structured data manipulation

# Visualization libraries
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns             # Advanced visualization (heatmaps, pairplots, etc.)

# Set visualization style
sns.set(style="whitegrid")

# Number of unique categories for each categorical value

def unique_values(df, sort=True):
    """
    Summarize the number of unique values per column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyze.
    sort : bool, optional (default=True)
        Whether to sort the output by number of unique values (descending).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'Column': column name
        - 'UniqueValues': number of unique (non-null) values
        - 'TotalValues': total number of rows
        - 'UniquePercent': % of unique values relative to total rows
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'UniqueValues': df.nunique(dropna=True).values,
        'TotalValues': len(df)
    })
    summary['UniquePercent'] = (summary['UniqueValues'] / summary['TotalValues'] * 100).round(2)

    if sort:
        summary = summary.sort_values(by='UniqueValues', ascending=False).reset_index(drop=True)

    return summary

# Plot Number of unique categories for each categorical value

def plot_number_of_unique_values(df, categorical_cols, dataset_name="Dataset", top_n=None):
    """
    Plots the number of unique values for categorical columns as BARCHART, with values displayed above each bar.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    categorical_cols : list
        List of categorical column names
    dataset_name : str
        Name of the dataset (for plot title)
    top_n : int, optional
        If specified, show only top_n features with most unique values
    """
    # Compute number of unique values per categorical column
    unique_counts = df[categorical_cols].nunique()
    
    if top_n is not None:
        unique_counts = unique_counts.sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(18, 6))
    bars = plt.bar(unique_counts.index, unique_counts.values, color='skyblue', edgecolor='black')
    
    # Annotate each bar with the number of unique values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{int(height)}', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.title(f"Number of Unique Values per Categorical Feature - {dataset_name}", 
              fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("Number of Unique Values", fontsize=12)
    plt.xlabel("Categorical Features", fontsize=12)
    plt.xticks(rotation=90)
    plt.ylim(0, max(unique_counts.values)*1.15)  # Add space for labels
    plt.tight_layout()
    plt.show()

    # Distributon for each categorical value

def plot_categorical_values_distributions(df, categorical_cols, dataset_name="Dataset", cols_per_row=4):
    """
    Plots the distribution of each categorical feature in a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing categorical features
    categorical_cols : list
        List of categorical column names
    dataset_name : str
        Name of the dataset (for the figure title)
    cols_per_row : int
        Number of subplots per row
    """
    n_cols = len(categorical_cols)
    n_rows = math.ceil(n_cols / cols_per_row)
    
    plt.figure(figsize=(cols_per_row*5, n_rows*4))
    plt.suptitle(f'Categorical Features Distribution - {dataset_name}', fontsize=18, fontweight='bold', y=1.02)
    
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(n_rows, cols_per_row, i)
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, color='mediumslateblue')
        plt.title(col, fontsize=12)
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.xlabel('')
    
    plt.tight_layout()
    plt.show()

    # Get column types

def get_column_types(df, verbose=True):
    """
    Returns lists of categorical, integer, and float columns from a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to inspect.
    verbose : bool, optional
        If True, prints the lists of columns. Default is True.

    Returns:
    --------
    object_cols : list
        List of categorical (object) columns.
    int_cols : list
        List of integer columns.
    float_cols : list
        List of float columns.
    """
    # Categorical columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Integer columns
    int_cols = df.select_dtypes(include=['int', 'int64']).columns.tolist()
    
    # Float columns
    float_cols = df.select_dtypes(include=['float', 'float64']).columns.tolist()
    
    # Print if verbose
    if verbose:
        print("Categorical variables:")
        print(object_cols)
        print("\nInteger variables:")
        print(int_cols)
        print("\nReal (float) variables:")
        print(float_cols)
    
    return object_cols, int_cols, float_cols