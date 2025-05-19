import pandas as pd
import numpy as np

# Constants for data processing
SLICE_START_INDEX = 1000
SLICE_END_INDEX = 91000  # Maximum value = 92500
MOVING_AVG_WINDOW = 20
MOVING_AVG_MODE = 'same' #'same', 'full'

def load_and_prepare_data(filepath, columns_to_use):
    """Loads data from a CSV file and selects relevant columns."""
    try:
        df = pd.read_csv(filepath)
        df = df[columns_to_use]
        return df
    except (FileNotFoundError, KeyError):
        return None

def clip_data(df, start_idx=SLICE_START_INDEX, end_idx=SLICE_END_INDEX):
    """Clips the dataframe rows from start_idx to end_idx (exclusive)."""
    return df.iloc[start_idx:end_idx]

def moving_average_filter(series, window_size=MOVING_AVG_WINDOW, mode=MOVING_AVG_MODE):
    """Applies moving average filter to a pandas Series."""
    if len(series) < window_size:
        return pd.Series(dtype=float)
    filtered = np.convolve(series, np.ones(window_size)/window_size, mode=MOVING_AVG_MODE)
    if MOVING_AVG_MODE == 'full':
        # Slice filtered array to original series length to match index
        filtered = filtered[:len(series)]
    return pd.Series(filtered, index=series.index)

def remove_dc_offset(series):
    """Removes DC offset from a pandas Series."""
    if series.empty:
        return series
    dc_offset = series.mean()
    return series - dc_offset
