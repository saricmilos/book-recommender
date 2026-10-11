# Core libraries
import math

# Data manipulation
import pandas as pd        # DataFrames for structured data manipulation
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns             # Advanced visualization (heatmaps, pairplots, etc.)

# Set visualization style
sns.set(style="whitegrid")

# Data Visualization

# Plot Missing Values HEATMAP

def missing_values_heatmap(df, dataset_name="Dataset"):
    """
    Plots a heatmap of missing values (HEATMAP) for a given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to visualize.
    dataset_name : str, optional
        Name of the dataset to be shown in the plot title (default = "Dataset").
    """
    plt.figure(figsize=(18, 6))
    sns.heatmap(
        df.isnull(),
        yticklabels=False,   # Hide row labels
        cbar=False,          # Remove color bar
        cmap="viridis"       # Colormap for missing values
    )

    plt.title(f"Heatmap of Missing Values - {dataset_name}", 
              fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Rows", fontsize=12)
    plt.show()

# Plot Missing Values BARCHART

def missing_values_barchart(df, dataset_name="Dataset", top_n=None, min_width=6):
    """
    Simple plot of missing-value percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    dataset_name : str
        Title suffix for the chart.
    top_n : int or None
        If set, show only the top_n features with the most missing values.
    min_width : float
        Minimum figure width in inches.
    """
    # percent missing per column
    missing_percent = df.isnull().mean() * 100

    # keep only columns with missing values > 0
    missing_percent = missing_percent[missing_percent > 0]

    # sort descending
    missing_percent = missing_percent.sort_values(ascending=False)

    if top_n is not None:
        missing_percent = missing_percent.head(top_n)

    if missing_percent.empty:
        print(f"No missing values in {dataset_name} dataset!")
        return missing_percent  # return empty Series for convenience

    # number of bars (you said you have only a few columns)
    n = len(missing_percent)

    # compute figure size: width scales with number of columns
    width = max(min_width, 1.0 * n)   # 1 inch per column is a reasonable default
    height = 4
    fig, ax = plt.subplots(figsize=(width, height))

    x = np.arange(n)
    vals = missing_percent.values

    ax.bar(x, vals, edgecolor='black', linewidth=0.6)

    # choose label format: show more decimals when values are very small
    max_val = vals.max()
    if max_val < 1.0:
        fmt = "{:.4f}%"
        text_offset = max_val * 0.05 + 0.0001
    else:
        fmt = "{:.1f}%"
        text_offset = max_val * 0.03 + 0.1

    # add labels above bars
    for xi, v in zip(x, vals):
        ax.text(xi, v + text_offset, fmt.format(v), ha='center', va='bottom', fontsize=9)

    # x ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(missing_percent.index.astype(str), rotation=45, ha='right', fontsize=9)

    ax.set_ylabel("Missing Values (%)")
    ax.set_title(f"Missing Values by Feature - {dataset_name}")

    # y limit with small margin so labels fit
    top = max_val * 1.2
    if top < 1.0:
        top = 1.0
    ax.set_ylim(0, top)

    # manual margins (keeps layout stable and avoids layout-engine conflicts)
    fig.subplots_adjust(bottom=0.28, top=0.92)

    plt.show()

    return missing_percent


# Missing Values Summary

def get_missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of missing values for each column in a DataFrame, including the column type.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to analyze.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with one row per column containing:
        - Column: Name of the column
        - Dtype: Data type of the column
        - TotalValues: Total number of rows in the DataFrame
        - MissingValues: Number of missing (NaN) values in the column
        - NonMissingValues: Number of non-missing values in the column
        - MissingPercent: Percentage of missing values in the column
    """
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [df[col].dtype for col in df.columns],
        "TotalValues": len(df),
        "MissingValues": df.isnull().sum().values,
        "NonMissingValues": df.notnull().sum().values
    })
    
    summary["MissingPercent"] = (summary["MissingValues"] / summary["TotalValues"]) * 100
    
    return summary

