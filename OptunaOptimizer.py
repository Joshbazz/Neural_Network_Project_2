import optuna
import pandas as pd
from keras.backend import clear_session
from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from fetch_data import fetch_fear_and_greed_btc
from DataPreprocessor import DataPreprocessor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

class HyperParameterOptimizer:
    def __init__(self, data, X_scaler=RobustScaler(), y_scaler=RobustScaler(), batch_size=64, validation_split=0.25, epochs=100):
        self.data = data
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.preprocessor = DataPreprocessor(X_scaler, y_scaler)

    def objective(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()

        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, _ = self.preprocessor.preprocess_data(self.data)

        # Print shapes for debugging
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1])) 
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        timesteps = X_train_scaled.shape[1] 
        features = X_train_scaled.shape[2]
        
        model = Sequential()
        model.add(Input(shape=(timesteps, features)))
        model.add(LSTM(units=trial.suggest_int('LSTM Neurons_0', 10, 100), return_sequences=True))
        model.add(Dropout(trial.suggest_float('Dropout Rate_0', 0.0001, 0.50)))
        model.add(Dense(trial.suggest_int('Dense Neurons', 1, 50), activation='relu'))
        model.add(Dense(1))  # No activation for regression

        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        model.compile(
            loss='mean_squared_error',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['mean_absolute_error']
        )
        
        model.fit(
            X_train_scaled,
            y_train_scaled,
            shuffle=False,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=False,
        )

        # Evaluate the model accuracy on the validation set.
        score = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        return score[1]

    def optimize(self, n_trials=5, timeout=100_000):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

# Example usage:
# if __name__ == "__main__":
#     data = fetch_fear_and_greed_btc()
#     X_scaler = RobustScaler()
#     y_scaler = RobustScaler()
    
#     optimizer = HyperParameterOptimizer(data, X_scaler, y_scaler)
#     optimizer.optimize(n_trials=5, timeout=100_000)
