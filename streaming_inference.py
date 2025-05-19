import numpy as np
import pandas as pd
import random
from tensorflow.keras.models import load_model
from data_processing import load_and_prepare_data, moving_average_filter, remove_dc_offset

FEATURE_COLUMNS = ['i_u', 'i_v', 'i_w', 'a_x', 'a_y', 'a_z']
SEGMENT_LENGTH = 256
MOVING_AVG_WINDOW = 20
TOTAL_SEGMENTS = 100

def preprocess_segment(segment_df):
    processed_cols = []
    for col in FEATURE_COLUMNS:
        filtered = moving_average_filter(segment_df[col], MOVING_AVG_WINDOW)
        filtered = filtered.dropna().reset_index(drop=True)
        dc_corrected = remove_dc_offset(filtered)
        processed_cols.append(dc_corrected)
    processed_df = pd.concat(processed_cols, axis=1)
    processed_df.columns = FEATURE_COLUMNS
    return processed_df

def get_random_segment(df):
    max_start = len(df) - SEGMENT_LENGTH
    start_idx = random.randint(0, max_start)
    segment = df.iloc[start_idx:start_idx + SEGMENT_LENGTH].reset_index(drop=True)
    return segment

def main():
    # Load datasets
    df_normal = load_and_prepare_data('without_load_10_5_2025.csv', FEATURE_COLUMNS)
    df_faulty = load_and_prepare_data('with_load_10_5_2025.csv', FEATURE_COLUMNS)
    if df_normal is None or df_faulty is None:
        print("Error loading data files.")
        return

    # Load model
    model = load_model('motor_fault_cnn_model.h5')

    # Define segment pattern: normal, normal, faulty, normal, faulty, faulty
    pattern = ['normal', 'normal', 'faulty', 'normal', 'faulty', 'faulty']
    pattern_len = len(pattern)

    for i in range(TOTAL_SEGMENTS):
        label = pattern[i % pattern_len]
        if label == 'normal':
            segment = get_random_segment(df_normal)
        else:
            segment = get_random_segment(df_faulty)

        # Preprocess segment
        processed_segment = preprocess_segment(segment)

        # No need to check length since moving average filter uses mode='same'

        # Prepare input for model: shape (1, SEGMENT_LENGTH, num_features)
        input_data = processed_segment.values.reshape(1, SEGMENT_LENGTH, len(FEATURE_COLUMNS))

        # Perform inference
        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        # Map predicted class to label
        predicted_label = 'faulty' if predicted_class == 1 else 'normal'

        # Print result
        print(f"Segment {i+1}: True Label = {label}, Predicted = {predicted_label}, Confidence = {confidence:.4f}")

if __name__ == '__main__':
    main()
