import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_categories(df, column, top_n=10, orientation='h', palette='pastel', show_percent=False, title=None):
    """
    Plot top N categories of a categorical column using a barplot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the column to plot
        column (str): Name of the categorical column
        top_n (int): Number of top categories to display
        orientation (str): 'h' for horizontal bars, 'v' for vertical bars
        palette (str or list): Seaborn color palette
        show_percent (bool): If True, shows percentages instead of counts
        title (str): Plot title
    """
    
    counts = df[column].value_counts().iloc[:top_n]
    
    if show_percent:
        total = len(df)
        counts = counts / total * 100
        xlabel = 'Percentage (%)'
    else:
        xlabel = 'Count'
    
    plt.figure(figsize=(10, 6))
    
    if orientation == 'h':
        sns.barplot(x=counts.values, y=counts.index, hue=counts.index, dodge=False, palette=palette, legend=False)
        plt.ylabel('')
        plt.xlabel(xlabel)
        # Annotate bars
        for i, value in enumerate(counts.values):
            plt.text(value + max(counts.values)*0.01, i,
                     f"{value:.1f}" if show_percent else f"{int(value)}",
                     va='center')
    else:
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, dodge=False, palette=palette, legend=False)
        plt.xlabel('')
        plt.ylabel(xlabel)
        plt.xticks(rotation=45, ha='right')
        # Annotate bars at the correct x positions (use category names)
        for i, (cat, value) in enumerate(zip(counts.index, counts.values)):
            plt.text(cat, value + max(counts.values)*0.01,
                     f"{value:.1f}" if show_percent else f"{int(value)}",
                     ha='center')
    
    if title is None:
        title = f"Top {top_n} {column.replace('_',' ').title()}"
    
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
def plot_boxplot(df, column, by=None, title=None, xlabel=None, ylabel=None,
                 palette='pastel', figsize=(8,6), rotate_xticks=False, show=True):
    """
    Plots a boxplot for a numeric column with optional grouping.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
        by (str, optional): Column name to group by (categorical).
        title (str, optional): Plot title.
        xlabel (str, optional): Label for x-axis.
        ylabel (str, optional): Label for y-axis.
        palette (str or list, optional): Seaborn color palette.
        figsize (tuple, optional): Figure size.
        rotate_xticks (bool, optional): Rotate x-axis tick labels if True.
        show (bool, optional): Whether to call plt.show().
        
    Returns:
        matplotlib.axes._subplots.AxesSubplot: The Axes object.
    """
    
    plt.figure(figsize=figsize)
    
    if by:
        ax = sns.boxplot(x=by, y=column, data=df, palette=palette)
    else:
        # single box: pick first color from palette
        ax = sns.boxplot(y=column, data=df, color=sns.color_palette(palette)[0])
    
    # Titles and labels
    if title:
        ax.set_title(title, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    
    # Rotate x-axis labels if needed
    if rotate_xticks and by:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return ax

def plot_histogram(
    df,
    column,
    bins=20,
    title=None,
    xlabel=None,
    ylabel='Count',
    figsize=(12,6),
    palette='Set2',
    top_n=None,
    by=None
):
    """
    Plots histograms for a numeric column, optionally grouped by top N categories.
    """
    df_plot = df.copy()
    
    if by and top_n:
        top_categories = df_plot[by].value_counts().nlargest(top_n).index
        df_plot = df_plot[df_plot[by].isin(top_categories)]
    
    if by and top_n:
        # FacetGrid to plot separate histogram for each category
        g = sns.FacetGrid(df_plot, col=by, col_wrap=5, height=4, sharex=True, sharey=True)
        g.map_dataframe(sns.histplot, column, bins=bins, color=sns.color_palette(palette)[0])
        g.set_axis_labels(xlabel or column, ylabel)
        g.fig.suptitle(title or f'{column} Distribution by {by}', fontweight='bold', y=1.05)
        plt.show()
    else:
        plt.figure(figsize=figsize)
        sns.histplot(df_plot[column], bins=bins, color=sns.color_palette(palette)[0], kde=False)
        plt.title(title or f'Distribution of {column}', fontweight='bold')
        plt.xlabel(xlabel or column)
        plt.ylabel(ylabel)
        plt.show()
