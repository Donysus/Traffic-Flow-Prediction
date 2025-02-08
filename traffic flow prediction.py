import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Dict
import joblib

# --------------------------
# Enhanced Data Generation
# --------------------------
def generate_traffic_data(n_points: int = 10080) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic traffic patterns with:
    - Daily cycles (1440 minutes = 24h * 60)
    - Weekly patterns
    - Random events (accidents, special events)
    - Luxury Vehicle traffic patterns (rush hours, factory shifts)
    """
    time = np.arange(n_points)

    # Base pattern (daily cycle)
    daily = 50 + 20 * np.sin(2 * np.pi * time / 1440)

    # Weekly pattern (5 work days + 2 weekend days)
    weekly = 10 * np.sin(2 * np.pi * time / 10080)  # 10080 = 7*1440

    # Patterns
    rush_hours = 15 * (np.exp(-(time % 1440 - 480)**2/(2*60**2)) +  # Morning rush (8:00)
                       np.exp(-(time % 1440 - 1020)**2/(2*60**2)))  # Evening rush (17:00)

    # Random events (1% probability of traffic incident)
    incidents = np.random.choice([0, 20], size=n_points, p=[0.99, 0.01])

    # Combine components with realistic noise
    traffic = daily + weekly + rush_hours + incidents + np.random.normal(0, 3, n_points)

    return time, np.clip(traffic, 0, 100)  # Clip to 0-100% capacity

# --------------------------
# Professional Data Preparation
# --------------------------
class TrafficPreprocessor:
    def __init__(self, window_size: int = 60, horizon: int = 12):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.window_size = window_size  # 60 steps = 5 hours (5-min intervals)
        self.horizon = horizon          # 1-hour prediction

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create time-series sequences with rolling window"""
        X, y = [], []
        for i in range(len(data) - self.window_size - self.horizon):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.horizon])
        return np.array(X), np.array(y)

    def preprocess(self, data: np.ndarray) -> Dict:
        """Full preprocessing pipeline"""
        # Differencing to remove trend
        differenced = np.diff(data, prepend=data[0])

        # Scaling
        scaled = self.scaler.fit_transform(differenced.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = self.create_sequences(scaled)

        return {
            'X': X[..., np.newaxis],  # Add channel dimension
            'y': y,
            'scaler': self.scaler
        }

# --------------------------
# Advanced LSTM Model
# --------------------------
def build_traffic_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """High-grade traffic prediction model"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True,
                         recurrent_dropout=0.2),
                      input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, recurrent_dropout=0.2)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(12)  # Predict next 12 steps (1 hour)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# --------------------------
# Training & Evaluation
# --------------------------
def train_evaluate_pipeline():
    # Generate and preprocess data
    time, traffic = generate_traffic_data()
    preprocessor = TrafficPreprocessor()
    processed = preprocessor.preprocess(traffic)

    # Split with time-series awareness
    split = int(len(processed['X']) * 0.8)
    X_train, y_train = processed['X'][:split], processed['y'][:split]
    X_test, y_test = processed['X'][split:], processed['y'][split:]

    # Build and train model
    model = build_traffic_model((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_pred = model.predict(X_test)
    test_pred = preprocessor.scaler.inverse_transform(test_pred)
    test_true = preprocessor.scaler.inverse_transform(y_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    mae = mean_absolute_error(test_true, test_pred)
    print(f"\nTest Performance:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")

    # Save artifacts
    model.save('mercedes_traffic_model.keras')
    joblib.dump(preprocessor.scaler, 'traffic_scaler.bin')

    return model, history, (test_true, test_pred)

# --------------------------
# Visualization & Analysis
# --------------------------
def plot_results(test_true: np.ndarray, test_pred: np.ndarray):
    plt.figure(figsize=(12, 6))

    # Plot 24-hour period
    plt.plot(test_true[:144], label='True Traffic', color='#2c3e50')
    plt.plot(test_pred[:144], label='Predicted', color='#e74c3c', linestyle='--')

    # Highlight peak hours
    for hour in [8, 17]:
        plt.axvspan(hour*12-12, hour*12, alpha=0.2, color='#f1c40f')

    plt.xlabel("Time (5-minute intervals)")
    plt.ylabel("Traffic Flow (%)")
    plt.title("Mercedes Traffic Flow Prediction Performance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------
# Execution
# --------------------------
if __name__ == "__main__":
    model, history, (test_true, test_pred) = train_evaluate_pipeline()
    plot_results(test_true, test_pred)
