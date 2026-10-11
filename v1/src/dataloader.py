# src/data_loader.py
from pathlib import Path
from typing import Any, Dict
import pandas as pd

# --------------------------------------
# Load a single CSV
# --------------------------------------
def load_dataset(csv_path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """
    Load a single CSV file into a pandas DataFrame.

    Args:
        csv_path (Path): Full path to the CSV file.
        **read_csv_kwargs: Additional keyword arguments to pass to `pd.read_csv`
            (e.g., `sep=',', index_col=0, parse_dates=True`).

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.ParserError: If the CSV cannot be parsed correctly.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path, **read_csv_kwargs)

# --------------------------------------
# Load all CSVs from a folder into a dictionary
# --------------------------------------
def load_all_csvs_from_folder(folder: Path, **read_csv_kwargs: Any) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files in a folder into a dictionary keyed by the file stem.

    Example:
        folder/
            train.csv
            test.csv
            extra.csv
        returns:
            {"train": df1, "test": df2, "extra": df3}

    Args:
        folder (Path | str): Path to the folder containing CSV files.
        **read_csv_kwargs: Additional keyword arguments for `pd.read_csv`.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping each CSV file name (without extension)
            to its corresponding DataFrame.

    Raises:
        FileNotFoundError: If the folder does not exist or is not a directory.
        pd.errors.ParserError: If any CSV cannot be parsed correctly.
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    datasets = {}
    for csv_file in folder.glob("*.csv"):
        datasets[csv_file.stem] = pd.read_csv(csv_file, **read_csv_kwargs)

    return datasets