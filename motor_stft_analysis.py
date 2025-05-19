import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
from data_processing import clip_data, moving_average_filter, remove_dc_offset, SLICE_START_INDEX, SLICE_END_INDEX, MOVING_AVG_WINDOW

def plot_stft_comparison(df_normal, df_unbalanced, parameters, image_folder, 
                         fs=45, nperseg=256, noverlap=128,
                         normal_label='Normal Load', unbalanced_label='Unbalanced Load'):
    """Plots STFT magnitude comparison for specified parameters."""
    for param in parameters:
        if param not in df_normal.columns or df_normal[param].empty or \
           param not in df_unbalanced.columns or df_unbalanced[param].empty:
            continue

        # Clip data
        normal_clipped = clip_data(df_normal)[param]
        unbalanced_clipped = clip_data(df_unbalanced)[param]

        # Filter data
        normal_filtered = moving_average_filter(normal_clipped, MOVING_AVG_WINDOW)
        unbalanced_filtered = moving_average_filter(unbalanced_clipped, MOVING_AVG_WINDOW)

        # Remove DC offset
        normal_processed = remove_dc_offset(normal_filtered)
        unbalanced_processed = remove_dc_offset(unbalanced_filtered)

        signal_normal = normal_processed.values
        signal_unbalanced = unbalanced_processed.values

        f_normal, t_normal, Zxx_normal = stft(signal_normal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        f_unbalanced, t_unbalanced, Zxx_unbalanced = stft(signal_unbalanced, fs=fs, nperseg=nperseg, noverlap=noverlap)

        fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        axs[0].pcolormesh(t_normal, f_normal, np.abs(Zxx_normal), shading='gouraud', cmap='viridis')
        axs[0].set_title(f'STFT Magnitude - {param} ({normal_label})')
        axs[0].set_ylabel('Frequency [cycles/sample]')
        axs[0].set_xlabel('Time [segments]')

        pcm = axs[1].pcolormesh(t_unbalanced, f_unbalanced, np.abs(Zxx_unbalanced), shading='gouraud', cmap='viridis')
        axs[1].set_title(f'STFT Magnitude - {param} ({unbalanced_label})')
        axs[1].set_xlabel('Time [segments]')

        max_time = max(t_normal[-1], t_unbalanced[-1])
        axs[0].set_xlim([0, max_time])
        axs[1].set_xlim([0, max_time])
        
        fig.colorbar(pcm, ax=axs[1], label='Magnitude')
        plt.tight_layout()
        
        filename = os.path.join(image_folder, f"stft_comparison_{param}_indices_{SLICE_START_INDEX}_{SLICE_END_INDEX-1}.png")
        plt.savefig(filename)
        plt.show()
