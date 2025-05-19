import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_processing import clip_data, moving_average_filter, remove_dc_offset, SLICE_START_INDEX, SLICE_END_INDEX

RMS_WINDOW_SIZE = 100

def plot_time_series_comparison(df_normal, df_unbalanced, params, image_folder, normal_label='Normal Load', unbalanced_label='Unbalanced Load'):
    """Plots time series comparison for given parameters."""
    print(f"\nGenerating time series plots for indices {SLICE_START_INDEX} to {SLICE_END_INDEX-1}...")

    for param in params:
        if param not in df_normal.columns or param not in df_unbalanced.columns:
            print(f"Warning: Parameter '{param}' not found in one or both DataFrames. Skipping.")
            continue

        # Clip data
        normal_clipped = clip_data(df_normal)[param]
        unbalanced_clipped = clip_data(df_unbalanced)[param]

        # Filter data
        normal_filtered = moving_average_filter(normal_clipped)
        unbalanced_filtered = moving_average_filter(unbalanced_clipped)

        plt.figure(figsize=(14, 7))
        plt.plot(normal_filtered.index, normal_filtered, label=f'{normal_label} - {param}', alpha=0.7)
        plt.plot(unbalanced_filtered.index, unbalanced_filtered, label=f'{unbalanced_label} - {param}', alpha=0.7)
        plt.xlabel('Index (Data Point Number)')
        plt.ylabel(f'{param} (Filtered)')
        plt.title(f'Time Series Comparison of {param} (Filtered)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(image_folder, f"timeseries_comparison_{param}_filtered_indices_{SLICE_START_INDEX}_{SLICE_END_INDEX-1}.png")
        plt.savefig(filename)
        plt.show()
    print("Time series plots generated.")

def plot_time_series_comparison_with_dc(df_normal, df_unbalanced, params, image_folder, normal_label='Normal Load', unbalanced_label='Unbalanced Load'):
    """Plots time series comparison for given parameters with DC offset removal."""
    print(f"\nGenerating time series plots with DC offset removal for indices {SLICE_START_INDEX} to {SLICE_END_INDEX-1}...")

    for param in params:
        if param not in df_normal.columns or param not in df_unbalanced.columns:
            print(f"Warning: Parameter '{param}' not found in one or both DataFrames. Skipping.")
            continue

        # Clip data
        normal_clipped = clip_data(df_normal)[param]
        unbalanced_clipped = clip_data(df_unbalanced)[param]

        # Filter data
        normal_filtered = moving_average_filter(normal_clipped)
        unbalanced_filtered = moving_average_filter(unbalanced_clipped)

        # Remove DC offset
        normal_processed = remove_dc_offset(normal_filtered)
        unbalanced_processed = remove_dc_offset(unbalanced_filtered)

        plt.figure(figsize=(14, 7))
        plt.plot(normal_processed.index, normal_processed, label=f'{normal_label} - {param}', alpha=0.7)
        plt.plot(unbalanced_processed.index, unbalanced_processed, label=f'{unbalanced_label} - {param}', alpha=0.7)
        plt.xlabel('Index (Data Point Number)')
        plt.ylabel(f'{param} (Filtered, DC Offset Corrected)')
        plt.title(f'Time Series Comparison of {param} (Filtered, DC Offset Corrected)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(image_folder, f"timeseries_comparison_{param}_filtered_dc_corrected_indices_{SLICE_START_INDEX}_{SLICE_END_INDEX-1}.png")
        plt.savefig(filename)
        plt.show()
    print("Time series plots with DC offset removal generated.")

def calculate_windowed_rms(series, window_size=RMS_WINDOW_SIZE):
    """Calculates RMS for non-overlapping windows of a series."""
    if len(series) < window_size:
        return pd.Series(dtype=float)
    rms_values = []
    for i in range(0, len(series) - window_size + 1, window_size):
        window = series[i:i+window_size]
        rms_val = (window**2).mean()**0.5
        rms_values.append(rms_val)
    return pd.Series(rms_values)

def plot_boxplot_comparison_rms(df_normal, df_unbalanced, params, image_folder, normal_label='Normal Load', unbalanced_label='Unbalanced Load', window_size=RMS_WINDOW_SIZE):
    """Plots boxplot comparison of windowed RMS for given parameters."""
    print(f"\nGenerating box plots of windowed RMS for indices {SLICE_START_INDEX} to {SLICE_END_INDEX-1}...")

    for param in params:
        if param not in df_normal.columns or param not in df_unbalanced.columns:
            print(f"Warning: Parameter '{param}' not found in one or both DataFrames. Skipping.")
            continue

        # Clip data
        normal_clipped = clip_data(df_normal)[param]
        unbalanced_clipped = clip_data(df_unbalanced)[param]

        # Filter and remove DC offset
        normal_processed = remove_dc_offset(moving_average_filter(normal_clipped))
        unbalanced_processed = remove_dc_offset(moving_average_filter(unbalanced_clipped))

        # Calculate windowed RMS
        normal_rms = calculate_windowed_rms(normal_processed, window_size)
        unbalanced_rms = calculate_windowed_rms(unbalanced_processed, window_size)

        if normal_rms.empty and unbalanced_rms.empty:
            print(f"Warning: Not enough data for RMS calculation for parameter '{param}'. Skipping.")
            continue

        # Prepare data for boxplot
        data_to_concat = []
        if not normal_rms.empty:
            data_to_concat.append(pd.DataFrame({'value': normal_rms, 'Load Condition': normal_label}))
        if not unbalanced_rms.empty:
            data_to_concat.append(pd.DataFrame({'value': unbalanced_rms, 'Load Condition': unbalanced_label}))

        combined_data = pd.concat(data_to_concat, ignore_index=True)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Load Condition', y='value', data=combined_data)
        plt.title(f'Box Plot of Windowed RMS for {param}')
        plt.ylabel(f'{param} (Windowed RMS, Window={window_size})')
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(image_folder, f"boxplot_comparison_{param}_rms_indices_{SLICE_START_INDEX}_{SLICE_END_INDEX-1}.png")
        plt.savefig(filename)
        plt.show()
    print("Box plots of windowed RMS generated.")

def plot_boxplot_comparison(df_normal, df_unbalanced, params, image_folder, normal_label='Normal Load', unbalanced_label='Unbalanced Load'):
    """Plots boxplot comparison for clipped data only (no filtering, no DC offset removal, no RMS calculation)."""
    print(f"\nGenerating box plots for clipped data only for indices {SLICE_START_INDEX} to {SLICE_END_INDEX-1}...")

    for param in params:
        if param not in df_normal.columns or param not in df_unbalanced.columns:
            print(f"Warning: Parameter '{param}' not found in one or both DataFrames. Skipping.")
            continue

        # Clip data only
        normal_clipped = clip_data(df_normal)[param]
        unbalanced_clipped = clip_data(df_unbalanced)[param]

        # Prepare data for boxplot
        data_to_concat = [
            pd.DataFrame({'value': normal_clipped, 'Load Condition': normal_label}),
            pd.DataFrame({'value': unbalanced_clipped, 'Load Condition': unbalanced_label})
        ]

        combined_data = pd.concat(data_to_concat, ignore_index=True)

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Load Condition', y='value', data=combined_data)
        plt.title(f'Box Plot Comparison for {param} (Clipped Data Only)')
        plt.ylabel(f'{param} (Clipped Data)')
        plt.grid(True)
        plt.tight_layout()
        filename = os.path.join(image_folder, f"boxplot_comparison_{param}_clipped_indices_{SLICE_START_INDEX}_{SLICE_END_INDEX-1}.png")
        plt.savefig(filename)
        plt.show()
    print("Box plots for clipped data generated.")
