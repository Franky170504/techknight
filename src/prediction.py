import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io

# --- MODULAR LOGIC FUNCTIONS ---

def load_data(file):
    """Loads data from a file-like object or path."""
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp')

def create_time_features(df):
    """Extracts cyclical time-based features from the timestamp."""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['quarter'] = df['timestamp'].dt.quarter
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    return df

def apply_one_hot_encoding(df, category_col='status'):
    """Converts categorical status into binary one-hot columns."""
    df_encoded = pd.get_dummies(df, columns=[category_col], prefix=category_col, dtype=int)
    status_cols = [col for col in df_encoded.columns if col.startswith(f"{category_col}_")]
    return df_encoded, status_cols

def train_and_evaluate(df, features, targets):
    """Trains individual XGBoost models for each target and calculates NRMSE."""
    models = {}
    metrics = []
    X = df[features]

    for target in targets:
        y = df[target]
        y_range = y.max() - y.min()
        if y_range == 0: y_range = 1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        models[target] = model
        
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        nrmse = rmse / y_range
        
        metrics.append({'Target': target, 'NRMSE': nrmse})
    
    return models, pd.DataFrame(metrics)

def generate_forecast(models, last_timestamp, features_list, status_cols, horizon_hours=24):
    """Generates future timestamps and predicts all target values."""
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq='h')
    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df = create_time_features(future_df)
    
    for target, model in models.items():
        future_df[target] = model.predict(future_df[features_list])
    
    # Reversing one-hot encoding back to classification
    future_df['predicted_status'] = future_df[status_cols].idxmax(axis=1)
    future_df['predicted_status'] = future_df['predicted_status'].str.replace('status_', '')
    
    return future_df

def run_full_pipeline(file):
    """Encapsulates the entire logic: Load -> Encode -> Train -> Forecast."""
    raw_data = load_data(file)
    data_with_time = create_time_features(raw_data)
    encoded_data, status_columns = apply_one_hot_encoding(data_with_time)
    
    # Define features and targets
    numerical_targets = ['water_level_m', 'rainfall_mm', 'temperature_c', 'demand_mcm', 'availability']
    features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    
    # Train and Forecast
    trained_models, _ = train_and_evaluate(encoded_data, features, numerical_targets + status_columns)
    forecast = generate_forecast(trained_models, raw_data['timestamp'].max(), features, status_columns)
    
    return forecast