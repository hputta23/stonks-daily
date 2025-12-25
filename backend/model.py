import numpy as np
import pandas as pd
# TensorFlow/Keras removed for lighter deployment
import datetime
from abc import ABC, abstractmethod

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
import math
from scipy.stats import norm

class BasePredictor(ABC):
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = ['Close', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI', 'MACD', 'Signal_Line', 'Upper_Band', 'Lower_Band']

    @abstractmethod
    def train(self, data, epochs=25, batch_size=32):
        pass

    @abstractmethod
    def predict_future(self, data, days=30):
        pass
        
    def tune_hyperparameters(self, x_train, y_train):
        # Default implementation: do nothing
        pass
        
    def backtest(self, data, split_ratio=0.8):
        # Default implementation for backtesting
        # 1. Split data
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # 2. Train on train_data
        self.train(train_data)
        
        # 3. Prepare test data for prediction
        # We need to prepare X_test such that it corresponds to y_test (test_data['Close'])
        # The prepare_data methods usually return X, y for the whole input.
        # So let's prepare for the WHOLE data, then slice the test part.
        
        # if isinstance(self, LSTMPredictor):
        #     X, y, scaled_data = self.prepare_data_lstm(data)
        # else:
        X, y, scaled_data = self.prepare_data_sklearn(data)
            
        # The 'y' array corresponds to data[look_back:].
        # We need to find the index in 'y' that corresponds to the start of test_data.
        # data index: 0 ... look_back ... train_size ... end
        # y index:    0 ... (train_size - look_back) ...
        
        # The first y value is at data.iloc[look_back].
        # We want y values starting from data.iloc[train_size].
        # So the index in y is train_size - look_back.
        
        test_start_idx = train_size - self.look_back
        
        if test_start_idx < 0:
            raise ValueError("Data too small for look_back window and split ratio")
            
        X_test = X[test_start_idx:]
        y_test = y[test_start_idx:]
        
        # 4. Predict
        # if isinstance(self, LSTMPredictor):
        #     predictions = self.model.predict(X_test, verbose=0)
        # else:
        predictions = self.model.predict(X_test)
        predictions = predictions.reshape(-1, 1)
            
        # 5. Inverse Transform
        # We only need to inverse transform the Close price (index 0)
        # Create dummy array for inverse transform
        dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
        dummy_array[:, 0] = predictions.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        dummy_array_y = np.zeros((len(y_test), len(self.feature_columns)))
        dummy_array_y[:, 0] = y_test.flatten()
        y_test_inv = self.scaler.inverse_transform(dummy_array_y)[:, 0]
        
        # 6. Metrics
        mae = mean_absolute_error(y_test_inv, predictions)
        rmse = math.sqrt(mean_squared_error(y_test_inv, predictions))
        r2 = r2_score(y_test_inv, predictions)
        mape = mean_absolute_percentage_error(y_test_inv, predictions)

        # Confidence Intervals (95%)
        residuals = y_test_inv - predictions
        std_dev = np.std(residuals)
        confidence_interval = 1.96 * std_dev
        
        # Risk Metrics (Forecast vs Actual Analysis)
        def calculate_risk_metrics(prices):
            if len(prices) < 2:
                return {}
            
            # Daily Returns
            returns = np.diff(prices) / prices[:-1]
            
            # Annualized Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe Ratio (assuming 0% risk free for simplicity)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            
            # VaR (95% Parametric)
            var_95 = norm.ppf(0.05, mean_return, std_return)
            
            # Max Drawdown
            cum_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / peak
            max_drawdown = np.min(drawdown)
            
            return {
                "volatility": volatility,
                "sharpe": sharpe_ratio,
                "var_95": var_95,
                "max_drawdown": max_drawdown
            }
            
        predicted_risk = calculate_risk_metrics(predictions)
        # We can also return 'actual' risk metrics if useful for comparison, 
        # but 'predicted_risk' is what characterizes the forecast's nature.
        
        # Feature Importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

        # 7. Dates
        dates = data['Date'].iloc[train_size:].values
        
        return {
            "dates": dates,
            "actual": y_test_inv.flatten(),
            "predicted": predictions.flatten(),
            "metrics": {
                "mae": mae, 
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "confidence_interval": confidence_interval,
                "volatility": predicted_risk.get("volatility", 0),
                "sharpe": predicted_risk.get("sharpe", 0),
                "var_95": predicted_risk.get("var_95", 0),
                "max_drawdown": predicted_risk.get("max_drawdown", 0)
            },
            "feature_importance": feature_importance
        }
        
    def prepare_data_lstm(self, data):
        # Select features
        dataset = data[self.feature_columns].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(dataset)
        
        x_train, y_train = [], []
        
        # We predict 'Close' (index 0) based on all features
        for i in range(self.look_back, len(scaled_data)):
            x_train.append(scaled_data[i-self.look_back:i, :]) # All features
            y_train.append(scaled_data[i, 0]) # Target is Close price (index 0)
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        
        # Reshape x_train for LSTM [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(self.feature_columns)))
        
        return x_train, y_train, scaled_data

    def prepare_data_sklearn(self, data):
        # For sklearn, we flatten the window
        dataset = data[self.feature_columns].values
        scaled_data = self.scaler.fit_transform(dataset)
        
        x_train, y_train = [], []
        
        for i in range(self.look_back, len(scaled_data)):
            # Flatten the window of features
            window = scaled_data[i-self.look_back:i, :]
            x_train.append(window.flatten())
            y_train.append(scaled_data[i, 0]) # Target is Close price
            
        return np.array(x_train), np.array(y_train), scaled_data

import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

class LSTMPredictor(BasePredictor):
    def build_model(self, input_shape):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is not installed. Cannot use LSTM predictor.")
            
        model = Sequential()
        model.add(Input(shape=input_shape))
        # Lightweight LSTM for memory efficiency
        model.add(LSTM(units=32, return_sequences=True)) # reduced from 50
        model.add(Dropout(0.2))
        model.add(LSTM(units=16, return_sequences=False)) # reduced from 50
        model.add(Dropout(0.2))
        model.add(Dense(units=16))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        
    def train(self, data, epochs=15, batch_size=32): # reduced epochs
        x_train, y_train, scaled_data = self.prepare_data_lstm(data)
        
        if self.model is None:
            self.build_model((x_train.shape[1], x_train.shape[2]))
            
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
        return history, scaled_data

    def predict_future(self, data, days=30):
        # Multivariate prediction logic (similar to RF but using 3D input)
        current_df = data.copy()
        future_dates = []
        predicted_prices = []
        last_date = current_df['Date'].iloc[-1]
        
        for i in range(days):
            dataset = current_df[self.feature_columns].values
            scaled_data = self.scaler.transform(dataset)
            
            # Get last window (3D for LSTM)
            current_batch = scaled_data[-self.look_back:].reshape(1, self.look_back, len(self.feature_columns))
            
            # Predict
            pred_scaled = self.model.predict(current_batch, verbose=0)[0][0]
            
            # Inverse transform
            dummy_row = np.zeros((1, len(self.feature_columns)))
            dummy_row[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predicted_prices.append(pred_price)
            
            # Next Date & Update DF
            next_date = last_date + datetime.timedelta(days=1)
            future_dates.append(next_date)
            last_date = next_date
            
            new_row = {'Date': next_date, 'Close': pred_price}
            new_df_row = pd.DataFrame([new_row])
            current_df = pd.concat([current_df, new_df_row], ignore_index=True)
            
            # Re-calc indicators (Simplified for brevity, assumes minimal drift in 1 step)
            # Ideally should duplicate logic from BasePredictor's other implementations
            # reusing code:
            self._update_indicators(current_df)
            current_df = current_df.fillna(method='ffill')
            
        return future_dates, np.array(predicted_prices)

    def _update_indicators(self, df):
        # Helper to update indicators on the growing dataframe
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Avoid division by zero
        loss = loss.replace(0, 0.001)
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        std_dev = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['SMA_20'] + (std_dev * 2)
        df['Lower_Band'] = df['SMA_20'] - (std_dev * 2)

class EnsemblePredictor(BasePredictor):
    def __init__(self, look_back=60):
        super().__init__(look_back)
        self.models = [
            RandomForestPredictor(look_back),
            GradientBoostingPredictor(look_back)
        ]
        
    def train(self, data, epochs=None, batch_size=None):
        for model in self.models:
            model.train(data, epochs, batch_size)
        
        class History:
            history = {'loss': [0]}
        return History(), None # Logic handled in sub-models

    def predict_future(self, data, days=30):
        # Gather predictions from all models
        all_preds = []
        future_dates = []
        for model in self.models:
            dates, preds = model.predict_future(data, days)
            all_preds.append(preds)
            future_dates = dates 
        
        # Average them
        avg_preds = np.mean(all_preds, axis=0)
        return future_dates, avg_preds
        
    def backtest(self, data, split_ratio=0.8):
        # Custom backtest for ensemble: average the backtest results of sub-models?
        # Or train and predict as a unit.
        # Let's train and predict as a unit using base implementation but overriding predict logic.
        # Actually base backtest calls 'train' then 'model.predict'.
        # Our 'model' attribute is None.
        # So we should override backtest to delegate properly or set a dummy model.
        # Simpler: Override backtest to run backtest on each submodel and average the 'predicted' array?
        # Yes, that's robust.
        
        results = []
        for model in self.models:
            res = model.backtest(data, split_ratio)
            results.append(res)
            
        # Combine
        dates = results[0]['dates']
        actual = results[0]['actual']
        
        # Average predictions
        preds_stack = np.vstack([r['predicted'] for r in results])
        avg_pred = np.mean(preds_stack, axis=0)
        
        # Recalculate metrics for the average
        mae = mean_absolute_error(actual, avg_pred)
        rmse = math.sqrt(mean_squared_error(actual, avg_pred))
        r2 = r2_score(actual, avg_pred)
        mape = mean_absolute_percentage_error(actual, avg_pred)
        
        # Risk Metrics
        # Need to import/define helper again or attach to self.
        # Let's just inline logic for now
        returns = np.diff(avg_pred) / avg_pred[:-1]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        mean_return = np.mean(returns) if len(returns) > 0 else 0
        std_return = np.std(returns) if len(returns) > 0 else 0
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        var_95 = norm.ppf(0.05, mean_return, std_return) if std_return > 0 else 0
        # Drawdown
        cum_ret = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak
        max_dd = np.min(dd) if len(dd) > 0 else 0

        # Feature Importance (average? specific to tree models)
        # Just take first model's for now or merge
        feat_imp = results[0]['feature_importance']
        
        return {
            "dates": dates,
            "actual": actual,
            "predicted": avg_pred,
            "metrics": {
                "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
                "volatility": volatility, "sharpe": sharpe_ratio, "var_95": var_95, "max_drawdown": max_dd
            },
            "feature_importance": feat_imp
        }           




class RandomForestPredictor(BasePredictor):
    def tune_hyperparameters(self, x_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
        search.fit(x_train, y_train)
        return search.best_estimator_

    def train(self, data, epochs=None, batch_size=None):
        x_train, y_train, scaled_data = self.prepare_data_sklearn(data)
        # self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model = self.tune_hyperparameters(x_train, y_train)
        self.model.fit(x_train, y_train)
        class History:
            history = {'loss': [0]}
        return History(), scaled_data

    def predict_future(self, data, days=30):
        # Multivariate prediction logic
        current_df = data.copy()
        future_dates = []
        predicted_prices = []
        last_date = current_df['Date'].iloc[-1]
        
        for i in range(days):
            dataset = current_df[self.feature_columns].values
            scaled_data = self.scaler.transform(dataset)
            
            # Get last window and flatten
            window = scaled_data[-self.look_back:, :]
            current_batch = window.flatten().reshape(1, -1)
            
            # Predict
            pred_scaled = self.model.predict(current_batch)[0]
            
            # Inverse transform
            dummy_row = np.zeros((1, len(self.feature_columns)))
            dummy_row[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predicted_prices.append(pred_price)
            
            # Next Date & Update DF
            next_date = last_date + datetime.timedelta(days=1)
            future_dates.append(next_date)
            last_date = next_date
            
            new_row = {'Date': next_date, 'Close': pred_price}
            new_df_row = pd.DataFrame([new_row])
            current_df = pd.concat([current_df, new_df_row], ignore_index=True)
            
            # Re-calc indicators
            current_df['SMA_20'] = current_df['Close'].rolling(window=20).mean()
            current_df['SMA_50'] = current_df['Close'].rolling(window=50).mean()
            current_df['EMA_12'] = current_df['Close'].ewm(span=12, adjust=False).mean()
            current_df['EMA_26'] = current_df['Close'].ewm(span=26, adjust=False).mean()
            
            delta = current_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            current_df['MACD'] = current_df['EMA_12'] - current_df['EMA_26']
            current_df['Signal_Line'] = current_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            std_dev = current_df['Close'].rolling(window=20).std()
            current_df['Upper_Band'] = current_df['SMA_20'] + (std_dev * 2)
            current_df['Lower_Band'] = current_df['SMA_20'] - (std_dev * 2)
            
            current_df = current_df.fillna(method='ffill')
        
        return future_dates, np.array(predicted_prices)

class SVRPredictor(BasePredictor):
    def tune_hyperparameters(self, x_train, y_train):
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.2]
        }
        search = RandomizedSearchCV(SVR(kernel='rbf'), param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
        search.fit(x_train, y_train)
        return search.best_estimator_

    def train(self, data, epochs=None, batch_size=None):
        x_train, y_train, scaled_data = self.prepare_data_sklearn(data)
        # self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        self.model = self.tune_hyperparameters(x_train, y_train)
        self.model.fit(x_train, y_train)
        class History:
            history = {'loss': [0]}
        return History(), scaled_data

    def predict_future(self, data, days=30):
        # Multivariate prediction logic
        current_df = data.copy()
        future_dates = []
        predicted_prices = []
        last_date = current_df['Date'].iloc[-1]
        
        for i in range(days):
            dataset = current_df[self.feature_columns].values
            scaled_data = self.scaler.transform(dataset)
            
            # Get last window and flatten
            window = scaled_data[-self.look_back:, :]
            current_batch = window.flatten().reshape(1, -1)
            
            # Predict
            pred_scaled = self.model.predict(current_batch)[0]
            
            # Inverse transform
            dummy_row = np.zeros((1, len(self.feature_columns)))
            dummy_row[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predicted_prices.append(pred_price)
            
            # Next Date & Update DF
            next_date = last_date + datetime.timedelta(days=1)
            future_dates.append(next_date)
            last_date = next_date
            
            new_row = {'Date': next_date, 'Close': pred_price}
            new_df_row = pd.DataFrame([new_row])
            current_df = pd.concat([current_df, new_df_row], ignore_index=True)
            
            # Re-calc indicators
            current_df['SMA_20'] = current_df['Close'].rolling(window=20).mean()
            current_df['SMA_50'] = current_df['Close'].rolling(window=50).mean()
            current_df['EMA_12'] = current_df['Close'].ewm(span=12, adjust=False).mean()
            current_df['EMA_26'] = current_df['Close'].ewm(span=26, adjust=False).mean()
            
            delta = current_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            current_df['MACD'] = current_df['EMA_12'] - current_df['EMA_26']
            current_df['Signal_Line'] = current_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            std_dev = current_df['Close'].rolling(window=20).std()
            current_df['Upper_Band'] = current_df['SMA_20'] + (std_dev * 2)
            current_df['Lower_Band'] = current_df['SMA_20'] - (std_dev * 2)
            
            current_df = current_df.fillna(method='ffill')
        
        return future_dates, np.array(predicted_prices)

class GradientBoostingPredictor(BasePredictor):
    def tune_hyperparameters(self, x_train, y_train):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42), param_grid, n_iter=5, cv=3, random_state=42, n_jobs=-1)
        search.fit(x_train, y_train)
        return search.best_estimator_

    def train(self, data, epochs=None, batch_size=None):
        x_train, y_train, scaled_data = self.prepare_data_sklearn(data)
        # self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        self.model = self.tune_hyperparameters(x_train, y_train)
        self.model.fit(x_train, y_train)
        class History:
            history = {'loss': [0]}
        return History(), scaled_data

    def predict_future(self, data, days=30):
        # Multivariate prediction logic
        current_df = data.copy()
        future_dates = []
        predicted_prices = []
        last_date = current_df['Date'].iloc[-1]
        
        for i in range(days):
            dataset = current_df[self.feature_columns].values
            scaled_data = self.scaler.transform(dataset)
            
            # Get last window and flatten
            window = scaled_data[-self.look_back:, :]
            current_batch = window.flatten().reshape(1, -1)
            
            # Predict
            pred_scaled = self.model.predict(current_batch)[0]
            
            # Inverse transform
            dummy_row = np.zeros((1, len(self.feature_columns)))
            dummy_row[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy_row)[0, 0]
            
            predicted_prices.append(pred_price)
            
            # Next Date & Update DF
            next_date = last_date + datetime.timedelta(days=1)
            future_dates.append(next_date)
            last_date = next_date
            
            new_row = {'Date': next_date, 'Close': pred_price}
            new_df_row = pd.DataFrame([new_row])
            current_df = pd.concat([current_df, new_df_row], ignore_index=True)
            
            # Re-calc indicators
            current_df['SMA_20'] = current_df['Close'].rolling(window=20).mean()
            current_df['SMA_50'] = current_df['Close'].rolling(window=50).mean()
            current_df['EMA_12'] = current_df['Close'].ewm(span=12, adjust=False).mean()
            current_df['EMA_26'] = current_df['Close'].ewm(span=26, adjust=False).mean()
            
            delta = current_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            current_df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            current_df['MACD'] = current_df['EMA_12'] - current_df['EMA_26']
            current_df['Signal_Line'] = current_df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            std_dev = current_df['Close'].rolling(window=20).std()
            current_df['Upper_Band'] = current_df['SMA_20'] + (std_dev * 2)
            current_df['Lower_Band'] = current_df['SMA_20'] - (std_dev * 2)
            
            current_df = current_df.fillna(method='ffill')
        
        return future_dates, np.array(predicted_prices)

class MonteCarloPredictor(BasePredictor):
    def train(self, data, epochs=None, batch_size=None):
        # Monte Carlo doesn't "train" in the traditional sense, 
        # but we calculate stats from historical data here.
        
        # Calculate Log Returns
        self.data = data.copy()
        self.data['Log_Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Calculate Drift and Volatility
        # We assume daily steps
        self.u = self.data['Log_Return'].mean()
        self.var = self.data['Log_Return'].var()
        self.drift = self.u - (0.5 * self.var)
        self.stdev = self.data['Log_Return'].std()
        
        class History:
            history = {'loss': [0]}
        return History(), None # No scaling needed for MC really, but interface expects it

    def backtest(self, data, split_ratio=0.8):
        # 1. Split data
        train_size = int(len(data) * split_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # 2. Train (calculate stats) on train_data
        self.train(train_data)
        
        # 3. Predict for test duration (treat steps as trading days)
        days = len(test_data)
        # Use GBM for backtesting baseline
        # We ignore the returned dates as we want to align with test_data dates
        _, mean_path, _ = self.predict_paths(train_data, days=days, iterations=1000, method="gbm")
        
        # 4. Metrics
        actual = test_data['Close'].values
        predicted = mean_path
        
        # Ensure lengths match (should be exact if steps=len(test_data))
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        dates = test_data['Date'].values[:min_len]
        
        mae = mean_absolute_error(actual, predicted)
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted)
        
        # Risk Metrics for MC (Predicted Path)
        returns = np.diff(predicted) / predicted[:-1]
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(252)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
            var_95 = norm.ppf(0.05, mean_return, std_return)
            cum_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / peak
            max_drawdown = np.min(drawdown)
        else:
             volatility = sharpe_ratio = var_95 = max_drawdown = 0

        return {
            "dates": dates,
            "actual": actual.tolist(),
            "predicted": predicted.tolist(),
            "metrics": {
                "mae": mae, 
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "volatility": volatility,
                "sharpe": sharpe_ratio,
                "var_95": var_95,
                "max_drawdown": max_drawdown
            }
        }

    def predict_future(self, data, days=30):
        # Generate simulations
        iterations = 1000
        last_price = data['Close'].iloc[-1]
        
        # Z values for all steps and iterations
        # shape: (days, iterations)
        Z = norm.ppf(np.random.rand(days, iterations))
        
        daily_returns = np.exp(self.drift + self.stdev * Z)
        
        # Price paths
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = last_price * daily_returns[0]
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
            
        # We return the MEAN path as the prediction
        predicted_prices = price_paths.mean(axis=1)
        
        # Dates
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=x+1) for x in range(days)]
        
        return future_dates, predicted_prices

    def predict_paths(self, data, days=30, iterations=50, method="gbm"):
        # Dispatch to specific simulation method
        if method == "jump_diffusion":
            return self._simulate_jump_diffusion(data, days, iterations)
        elif method == "heston":
            return self._simulate_heston(data, days, iterations)
        elif method == "bootstrapping":
            return self._simulate_bootstrap(data, days, iterations)
        elif method == "ou":
            return self._simulate_ou(data, days, iterations)
        elif method == "cev":
            return self._simulate_cev(data, days, iterations)
        else:
            # Default to GBM for unimplemented models (with a warning if possible, but for now fallback)
            return self._simulate_gbm(data, days, iterations)

    def _simulate_gbm(self, data, days, iterations):
        last_price = data['Close'].iloc[-1]
        
        # Z values for all steps and iterations
        Z = norm.ppf(np.random.rand(days, iterations))
        
        daily_returns = np.exp(self.drift + self.stdev * Z)
        
        # Price paths
        price_paths = np.zeros_like(daily_returns)
        price_paths[0] = last_price * daily_returns[0]
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * daily_returns[t]
            
        return self._format_output(data, price_paths, days)

    def _simulate_jump_diffusion(self, data, days, iterations):
        # Merton Jump Diffusion Model
        # Estimate parameters from historical data
        log_returns = self.data['Log_Return'].dropna()
        mean_return = log_returns.mean()
        std_return = log_returns.std()
        
        # Identify jumps as returns > 3 std devs
        jumps = log_returns[np.abs(log_returns - mean_return) > 3 * std_return]
        
        mu = self.drift
        sigma = self.stdev
        
        if len(jumps) > 0:
            lambda_jump = len(jumps) / len(log_returns) # Daily jump probability
            mu_jump = jumps.mean()
            sigma_jump = jumps.std() if len(jumps) > 1 else std_return * 2
        else:
            # Fallback if no jumps detected in history
            lambda_jump = 0.01 
            mu_jump = 0.0
            sigma_jump = std_return * 3
        
        last_price = data['Close'].iloc[-1]
        dt = 1
        
        price_paths = np.zeros((days, iterations))
        price_paths[0] = last_price
        
        for t in range(1, days):
            # GBM component
            Z = np.random.standard_normal(iterations)
            gbm_return = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            
            # Jump component
            J = np.random.poisson(lambda_jump * dt, iterations)
            jump_multiplier = np.ones(iterations)
            if np.any(J > 0):
                # For each jump, simulate size
                # We simplify and apply one aggregate jump if J > 0
                # Or loop. Vectorized approach:
                # If J=1, multiply by lognormal. If J=0, multiply by 1.
                # We'll assume max 1 jump per step for simplicity or sum log returns
                jump_sizes = np.random.normal(mu_jump, sigma_jump, iterations)
                # Apply jump only where J > 0
                jump_multiplier = np.where(J > 0, np.exp(jump_sizes), 1.0)
            
            price_paths[t] = price_paths[t-1] * gbm_return * jump_multiplier
            
        return self._format_output(data, price_paths, days)

    def _simulate_heston(self, data, days, iterations):
        # Heston Stochastic Volatility Model
        # Parameters
        mu = self.drift # Drift
        v0 = self.stdev ** 2 # Initial variance
        kappa = 2.0 # Mean reversion speed
        theta = v0 # Long run variance
        xi = 0.3 # Volatility of volatility
        rho = -0.7 # Correlation between price and volatility
        
        last_price = data['Close'].iloc[-1]
        dt = 1/252 # Time step (assuming daily steps in a trading year)
        # Adjust drift for daily step
        mu_annual = mu * 252
        
        price_paths = np.zeros((days, iterations))
        price_paths[0] = last_price
        
        vt = np.full(iterations, v0)
        st = np.full(iterations, last_price)
        
        for t in range(1, days):
            # Correlated Brownian Motions
            Z1 = np.random.standard_normal(iterations)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(iterations)
            
            # Update Volatility (ensure non-negative)
            # dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
            vt = np.maximum(0, vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * np.sqrt(dt) * Z2)
            
            # Update Price
            # dS = mu * S * dt + sqrt(v) * S * dW1
            st = st * np.exp((mu_annual - 0.5 * vt) * dt + np.sqrt(vt) * np.sqrt(dt) * Z1)
            
            price_paths[t] = st
            
        return self._format_output(data, price_paths, days)

    def _simulate_bootstrap(self, data, days, iterations):
        # Historical Bootstrapping (Resampling)
        log_returns = self.data['Log_Return'].dropna().values
        last_price = data['Close'].iloc[-1]
        
        # Randomly sample returns with replacement
        random_returns = np.random.choice(log_returns, size=(days, iterations))
        
        price_paths = np.zeros((days, iterations))
        price_paths[0] = last_price * np.exp(random_returns[0])
        
        for t in range(1, days):
            price_paths[t] = price_paths[t-1] * np.exp(random_returns[t])
            
        return self._format_output(data, price_paths, days)

    def _simulate_ou(self, data, days, iterations):
        # Ornstein-Uhlenbeck Process (Mean Reverting)
        # dX = theta * (mu - X) * dt + sigma * dW
        
        last_price = data['Close'].iloc[-1]
        
        # Estimate parameters
        # We assume the price itself is mean reverting (simple view)
        # Or returns are mean reverting. Let's model Price for visual distinctness.
        mu = data['Close'].mean() # Long term mean price
        sigma = data['Close'].std() * 0.5 # Volatility
        theta = 0.1 # Mean reversion speed
        
        dt = 1
        price_paths = np.zeros((days, iterations))
        price_paths[0] = last_price
        
        for t in range(1, days):
            Z = np.random.standard_normal(iterations)
            # Euler-Maruyama
            dx = theta * (mu - price_paths[t-1]) * dt + sigma * np.sqrt(dt) * Z
            price_paths[t] = price_paths[t-1] + dx
            # Ensure non-negative
            price_paths[t] = np.maximum(price_paths[t], 0.01)
            
        return self._format_output(data, price_paths, days)

    def _simulate_cev(self, data, days, iterations):
        # Constant Elasticity of Variance (CEV)
        # dS = mu * S * dt + sigma * S^gamma * dW
        
        last_price = data['Close'].iloc[-1]
        mu = self.drift
        sigma = self.stdev
        gamma = 0.8 # Elasticity parameter (gamma < 1 for leverage effect)
        
        dt = 1
        price_paths = np.zeros((days, iterations))
        price_paths[0] = last_price
        
        for t in range(1, days):
            Z = np.random.standard_normal(iterations)
            # S^gamma
            s_gamma = np.power(np.maximum(price_paths[t-1], 0.001), gamma)
            
            ds = mu * price_paths[t-1] * dt + sigma * s_gamma * np.sqrt(dt) * Z
            price_paths[t] = price_paths[t-1] + ds
            price_paths[t] = np.maximum(price_paths[t], 0.01)
            
        return self._format_output(data, price_paths, days)

    def _format_output(self, data, price_paths, days):
        # Calculate mean path
        mean_path = price_paths.mean(axis=1)
        
        # Dates
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=x+1) for x in range(days)]
        
        # Transpose price_paths to be list of paths (iterations, days)
        paths_list = price_paths.T.tolist()
        
        return future_dates, mean_path, paths_list

def get_predictor(model_type: str):
    if model_type == "lstm":
        return LSTMPredictor()
    elif model_type == "random_forest":
        return RandomForestPredictor()
    elif model_type == "svr":
        return SVRPredictor()
    elif model_type == "gradient_boosting":
        return GradientBoostingPredictor()
    elif model_type == "monte_carlo":
        return MonteCarloPredictor()
    elif model_type == "ensemble":
        return EnsemblePredictor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
