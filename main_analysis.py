import os
import pandas as pd
from data_processing import load_and_prepare_data
import motor_time_analysis
import motor_stft_analysis

def main():
    # File paths
    file_unbalanced = 'with_load_10_5_2025.csv'
    file_normal = 'without_load_10_5_2025.csv'

    # Image output folder
    image_folder = 'image'
    os.makedirs(image_folder, exist_ok=True)

    # Parameters for different analyses
    time_series_params = ['v_u', 'v_v', 'v_w', 'rpm', 'temp']
    current_accel_params = ['i_u', 'i_v', 'i_w', 'a_x', 'a_y', 'a_z']

    # Load data
    print("Loading data...")
    df_unbalanced = pd.read_csv(file_unbalanced)
    df_normal = pd.read_csv(file_normal)

    # Check if data loaded successfully
    if df_unbalanced.empty or df_normal.empty:
        print("Error: One or both CSV files are empty or could not be loaded.")
        return

    # Plot time series comparison without DC offset removal
    motor_time_analysis.plot_time_series_comparison(
        df_normal, df_unbalanced, time_series_params, image_folder
    )

    # Plot boxplot comparison
    motor_time_analysis.plot_boxplot_comparison(
        df_normal, df_unbalanced, time_series_params, image_folder
    )

    # Plot time series comparison with DC offset removal
    motor_time_analysis.plot_time_series_comparison_with_dc(
        df_normal, df_unbalanced, current_accel_params, image_folder
    )

    # Plot boxplot comparison of windowed RMS
    motor_time_analysis.plot_boxplot_comparison_rms(
        df_normal, df_unbalanced, current_accel_params, image_folder
    )

    # Plot STFT comparison
    motor_stft_analysis.plot_stft_comparison(
        df_normal, df_unbalanced, current_accel_params, image_folder
    )

if __name__ == "__main__":
    main()
