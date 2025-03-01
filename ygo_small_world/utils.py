import pandas as pd

def sub_df(df: pd.DataFrame, column_values: list, column_name: str) -> pd.DataFrame:
    """
    Utility function. Creates a subset of the given DataFrame based on specified values in a particular column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame from which the subset will be extracted.
        column_values (list): A list of values to match against the specified column to filter rows.
        column_name (str): The name of the column in which to look for the specified values.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where the specified column contains any of the values in 'column_values'.
    """
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a valid column in the DataFrame.")

    if not pd.Series(column_values).isin(df[column_name]).any():
        raise ValueError("No values are in df. Data may need to be updated.")

    mask = df[column_name].isin(column_values)
    return df.loc[mask].copy()
