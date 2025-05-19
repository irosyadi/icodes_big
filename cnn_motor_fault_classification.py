import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from data_processing import load_and_prepare_data, clip_data, moving_average_filter, remove_dc_offset

# Constants
FEATURE_COLUMNS = ['i_u', 'i_v', 'i_w', 'a_x', 'a_y', 'a_z']
SEQUENCE_LENGTH = 256
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
MOVING_AVG_WINDOW = 20
MOVING_AVG_MODE = 'same' #'same', 'full'
CLIP_START = 1000
CLIP_END = 91000

def preprocess_dataset(filepath):
    # Load and select relevant columns
    df = load_and_prepare_data(filepath, FEATURE_COLUMNS)
    if df is None:
        raise FileNotFoundError(f"File {filepath} not found or missing columns.")
    # Clip data
    df_clipped = clip_data(df, CLIP_START, CLIP_END)
    # Apply moving average filter and remove DC offset for each column
    processed_cols = []
    for col in FEATURE_COLUMNS:
        filtered = moving_average_filter(df_clipped[col], MOVING_AVG_WINDOW, MOVING_AVG_MODE)
        filtered = filtered.dropna().reset_index(drop=True)
        dc_corrected = remove_dc_offset(filtered)
        processed_cols.append(dc_corrected)
    # Combine processed columns into DataFrame
    processed_df = pd.concat(processed_cols, axis=1)
    processed_df.columns = FEATURE_COLUMNS
    return processed_df

def create_sequences(data, seq_length=SEQUENCE_LENGTH):
    # Create sequences of shape (num_samples, seq_length, num_features)
    num_features = data.shape[1]
    total_length = data.shape[0]
    sequences = []
    for start_idx in range(0, total_length - seq_length + 1, seq_length):
        seq = data.iloc[start_idx:start_idx + seq_length].values
        if seq.shape[0] == seq_length:
            sequences.append(seq)
    return np.array(sequences)

def prepare_data():
    # Preprocess both datasets
    data_unbalanced = preprocess_dataset('with_load_10_5_2025.csv')
    data_normal = preprocess_dataset('without_load_10_5_2025.csv')

    # Create sequences
    seq_unbalanced = create_sequences(data_unbalanced)
    seq_normal = create_sequences(data_normal)

    # Create labels: 1 for unbalanced, 0 for normal
    labels_unbalanced = np.ones(len(seq_unbalanced))
    labels_normal = np.zeros(len(seq_normal))

    # Combine data and labels
    X = np.concatenate((seq_unbalanced, seq_normal), axis=0)
    y = np.concatenate((labels_unbalanced, labels_normal), axis=0)

    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into train, val, test
    train_size = int(len(X) * TRAIN_RATIO)
    val_size = int(len(X) * VAL_RATIO)
    test_size = len(X) - train_size - val_size

    X_train, X_temp = X[:train_size], X[train_size:]
    y_train, y_temp = y[:train_size], y[train_size:]

    X_val, X_test = X_temp[:val_size], X_temp[val_size:]
    y_val, y_test = y_temp[:val_size], y_temp[val_size:]

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate():
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=2
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save the model
    model.save('motor_fault_cnn_model.h5')
    print("Model saved as motor_fault_cnn_model.h5")

if __name__ == "__main__":
    train_and_evaluate()
