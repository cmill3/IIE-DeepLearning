import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import warnings 
import math
warnings.filterwarnings('ignore')


def debug_print(tensor, name):
    print(f"Debug {name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
    print(f"Sample values:\n{tensor[0, :5, :5]}\n")

def create_sequences(data, seq_length, prediction_horizon, features):
    symbols = data['symbol_encoded'].unique()
    xs = []
    symbol_seq = []
    for symbol in symbols:
        symbol_data = data[data['symbol_encoded'] == symbol].reset_index(drop=True)
        # symbol_data = symbol_data[features]
        symbol_data_values = symbol_data.values
        for i in range(len(symbol_data_values) - seq_length - prediction_horizon + 1):
            x = symbol_data_values[i:(i + seq_length)]
            symbol_seq.append(symbol)
            xs.append(x)
    return np.array(xs), symbol_seq

def create_target(data, seq_length, prediction_horizon, target="target"):
    symbols = data['symbol_encoded'].unique()
    ys = []
    for symbol in symbols:
        symbol_data = data[data['symbol_encoded'] == symbol].reset_index(drop=True)
        target_df = symbol_data[target]
        target_values = target_df.values
        for i in range(len(target_values) - seq_length - prediction_horizon + 1):
            y = target_values[i + seq_length + prediction_horizon - 1]
            ys.append(y)
    return np.array(ys)

def create_target_sequences(data, seq_length, prediction_horizon, features):
    symbols = data['symbol_encoded'].unique()
    ys = []
    symbol_seq = []
    for symbol in symbols:
        symbol_data = data[data['symbol_encoded'] == symbol].reset_index(drop=True)
        # symbol_data = symbol_data[features]
        symbol_data_values = symbol_data.values
        for i in range(len(symbol_data_values) - seq_length - prediction_horizon + 1):
            y = symbol_data_values[i + seq_length:i + seq_length + prediction_horizon]
            ys.append(y)
            symbol_seq.append(symbol)
    return np.array(ys), symbol_seq


def create_trend_sequences(ts_data, seq_length, prediction_horizon, model_hyperparameters, alerts):
    xs = []
    xs_cat = []
    ys = []
    symbols = alerts['symbol'].unique()
    for symbol in symbols:
        symbol_alerts = alerts.loc[alerts['symbol'] == symbol].reset_index(drop=True)
        symbol_df = ts_data.loc[ts_data['symbol'] == symbol].reset_index(drop=True)
        symbol_df['idx_value'] = symbol_df.index
        for idx, row in symbol_alerts.iterrows():
            try:
                match_dt = f"{row['date']} {row['hour']}:{row['minute']}"
                position = symbol_df[symbol_df['dt'] == match_dt]['idx_value'].values[0]

                instance_start = (position - seq_length + 1)
                target_end = position + prediction_horizon
                symbol_df['upside_threshold_prediction'] = row['upside_threshold_prediction']
                symbol_df['downside_threshold_prediction'] = row['downside_threshold_prediction']
                # scaler = TemporalFeatureScaler(window_size=model_hyperparameters['context_length'], min_periods=2, scaling_method='rolling')
                # symbol = ts_data[ts_data['symbol'] == row['symbol']].reset_index(drop=True)
                # num_scaled = scaler.fit_transform(group_data[num_features])
                X = symbol_df[model_hyperparameters['numerical_features']].values[instance_start:(position+1)]
                X_cat = symbol_df[model_hyperparameters['categorical_features']].values[instance_start:(position+1)]
                Y = symbol_df[model_hyperparameters['numerical_features']].values[(position+1):(target_end+1)]

                if len(X) != seq_length or len(Y) != prediction_horizon:
                    # print(f"Error in creating trend sequences: {len(X)} {len(Y)}")
                    # print(f"Symbol: {symbol_df}")
                    # print(match_dt)
                    # print(symbol)
                    # print(row)
                    continue

                xs.append(X.astype(np.float32))
                xs_cat.append(X_cat.astype(np.int8))
                ys.append(Y.astype(np.float32))
            except Exception as e:
                # print(f"Error in creating trend sequences: {e} for {row['date']}")
                # print(symbol_df.loc[symbol_df['ymd'] == row['date']])
                # print(f"Error in creating trend sequences: {e}")
                # print(match_dt)
                # print(symbol)
                ## remove row from symbol alerts
                symbol_alerts.drop(idx,inplace=True)
                continue

    try:
        x = np.array(xs, dtype=np.float32)
        cat = np.array(xs_cat, dtype=np.int64)
        y = np.array(ys, dtype=np.float32)
    except Exception as e:
        print(f"Error in creating trend sequences: {e}")
        print(xs)
    return x, cat, y


def create_trend_sequences_test(ts_data, seq_length, prediction_horizon, model_hyperparameters, alerts, unscaled_df):
    xs = []
    xs_cat = []
    ys = []
    ys_target = []
    completed_alerts = []
    for _, row in alerts.iterrows():
        try:
            symbol_df = ts_data.loc[ts_data['symbol'] == row['symbol']].reset_index(drop=True)
            unscaled_symbol_df = unscaled_df.loc[unscaled_df['symbol'] == row['symbol']].reset_index(drop=True)
            symbol_df['idx_value'] = symbol_df.index
            match_dt = f"{row['date']} {row['hour']}:{row['minute']}"
            position = symbol_df[symbol_df['dt'] == match_dt]['idx_value'].values[0]

            instance_start = position - seq_length
            target_end = position + prediction_horizon
            symbol_df['upside_threshold_prediction'] = row['upside_threshold_prediction']
            symbol_df['downside_threshold_prediction'] = row['downside_threshold_prediction']
            # scaler = TemporalFeatureScaler(window_size=model_hyperparameters['context_length'], min_periods=2, scaling_method='rolling')
            # symbol = ts_data[ts_data['symbol'] == row['symbol']].reset_index(drop=True)
            # num_scaled = scaler.fit_transform(group_data[num_features])
            X = symbol_df[model_hyperparameters['numerical_features']].values[instance_start:position]
            X_cat = symbol_df[model_hyperparameters['categorical_features']].values[instance_start:position]
            Y = symbol_df[model_hyperparameters['numerical_features']].values[position:target_end]
            Y_tg = unscaled_symbol_df[model_hyperparameters['numerical_features']].values[position:target_end]

            xs.append(X)
            xs_cat.append(X_cat)
            ys.append(Y)
            ys_target.append(Y_tg)
            completed_alerts.append(row)
        except Exception as e:
            print(f"Error in creating trend sequences: {e}")
            # print(f"Symbol: {symbol_df}")
            # print(match_dt)
            # print(row)
            continue
    new_alerts_df = pd.DataFrame.from_records(completed_alerts)
    return np.array(xs), np.array(xs_cat), np.array(ys), np.array(ys_target), new_alerts_df

def create_scaled_sequences(dataset, num_features, categorical_features, model_hyperparameters, other_columns):
    print("Creating trend sequences")
    group_ids = dataset['group_id'].unique()
    data = []
    for index, group_id in enumerate(group_ids):
        print(f"Group {index+1}/{len(group_ids)}")
        scaler = TemporalFeatureScaler(window_size=model_hyperparameters['context_length'], min_periods=10, scaling_method='rolling')
        group_data = dataset.loc[dataset['group_id'] == group_id].reset_index(drop=True)
        num_scaled = scaler.fit_transform(group_data[num_features])
        X = num_scaled.values[10:]
        X_cat = group_data[other_columns].values[10:]
        x_df = pd.DataFrame(X, columns=num_features)
        cat_df = pd.DataFrame(X_cat, columns=other_columns)

        full_df = pd.concat([x_df, cat_df], axis=1)
        # full_df['group_id'] = group_id


        data.append(full_df)

    scaled_df = pd.concat(data)
    return scaled_df

# def prepare_scaled_training_data(scaled_data, categorical_data, seq_length, prediction_horizon):
#     group_data = scaled_data[scaled_data['group_ids'] == group_id].reset_index(drop=True)
#     group_values = group_data['target'].values
#     num_data = group_data[num_features].values
#     cat_data = group_data[categorical_features].values
#     x = num_data[-(seq_length+prediction_horizon):]
#     x_cat = cat_data[-(seq_length+prediction_horizon):]
#     xs.append(x)
#     xs_cat.append(x_cat)
#     return np.array(xs), np.array(xs_cat)


class TemporalFeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=100, min_periods=10, scaling_method='rolling', prediction_horizon=8):
        self.window_size = window_size
        self.min_periods = min_periods
        self.scaling_method = scaling_method
        self.prediction_horizon = prediction_horizon
        self.feature_scalers = {}

    def fit(self, X, y=None):
        # X should be a pandas DataFrame
        for column in X.columns:
            if self.scaling_method == 'expanding':
                self.feature_scalers[column] = {
                    'mean': X[column].expanding(min_periods=self.min_periods).mean(),
                    'std': X[column].expanding(min_periods=self.min_periods).std()
                }
            elif self.scaling_method == 'rolling':
                ## double check if this is correct
                self.feature_scalers[column] = {
                    'mean': X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean(),
                    'std': X[column].rolling(window=self.window_size, min_periods=self.min_periods).std()
                }
        return self

    def transform(self, X):
        X_scaled = X.copy()
        for column in X.columns:
            if column in self.feature_scalers:
                mean = self.feature_scalers[column]['mean']
                std = self.feature_scalers[column]['std']
                
                # Update rolling statistics
                # if self.scaling_method == 'rolling':
                #     mean = mean.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).mean())
                #     std = std.append(X[column].rolling(window=self.window_size, min_periods=self.min_periods).std())
                #     self.feature_scalers[column]['mean'] = mean
                #     self.feature_scalers[column]['std'] = std
                
                # Apply normalization using the updated statistics
                X_scaled[column] = (X[column] - mean) / (std + 1e-8)
                
                # Handle NaNs
                X_scaled[column] = X_scaled[column].fillna(method='ffill').fillna(0)
        
        return X_scaled


    def inverse_transform(self, X, evaluation_column, alert_index, model_hyperparameters):
        X_inverse = X.cpu().detach().numpy()
        if evaluation_column == 'h':
            target = X_inverse[:, 1]
        elif evaluation_column == 'l':
            target = X_inverse[:, 2]

        # try:
        mean = self.feature_scalers[evaluation_column]['mean'].reset_index(drop=True)
        std = self.feature_scalers[evaluation_column]['std'].reset_index(drop=True)
        # instance_start = alert_index - model_hyperparameters['context_length']
        instance_std = std[alert_index:alert_index + model_hyperparameters['prediction_horizon']]
        instance_mean = mean[alert_index:alert_index + model_hyperparameters['prediction_horizon']]

        unscaled_target = (target * instance_std) + instance_mean
        # except:
        #     print(f"Error in inverse transform for column {column}")
        return unscaled_target
    

def build_scaled_datasets(file_path, model_hyperparameters):
        print("Building scaled data")
        dataset = pd.read_csv(file_path)
        # dataset = dataset.loc[dataset['symbol'].isin(['AAPL','SPY','QQQ','AMD','NVDA','AMZN'])]
        dataset = dataset.loc[(dataset['hour'] >= 9) & (dataset['hour'] <= 16)]
        dataset['v_diff'] = dataset['v'].pct_change()
        dataset['v_diff'] = dataset['v_diff'].fillna(0)

        # Fit and transform the 'category' column
        dataset['dow'] = pd.to_datetime(dataset['day']).dt.dayofweek
        dataset['dom'] = pd.to_datetime(dataset['day']).dt.day
        dataset['year'] = pd.to_datetime(dataset['day']).dt.year

        # Bottom out year and hour so they fit into their representative embedding space
        dataset['year'] = dataset['year'] - dataset['year'].min()
        dataset['hour'] = dataset['hour'] - dataset['hour'].min()
        dataset['bb_category'] = dataset['bb_category'] + 1

        numerical_columns = model_hyperparameters['numerical_features']
        other_columns = dataset.columns.difference(numerical_columns)
        print(other_columns)
        scaled_data = create_scaled_sequences(dataset, numerical_columns, model_hyperparameters['categorical_features'], model_hyperparameters, other_columns)
        return scaled_data

def build_market_datasets_categorical(ts_data, model_hyperparameters, alerts_df):
    print("Building market datasets CAT")
    # dataset = pd.read_csv(file_path)
    # dataset = dataset.loc[dataset['symbol'].isin(['AAPL','SPY','QQQ','AMD','NVDA','AMZN'])]
    # dataset['period_volatility'] = dataset['h'] - dataset['l']/dataset['c']
    # dataset['oc_diff'] = dataset['o'] - dataset['c']/dataset['c']
    # dataset = dataset.loc[(dataset['hour'] >= 9) & (dataset['hour'] <= 16)]
    # dataset['v_diff'] = dataset['v'].pct_change()
    # dataset['v_diff'] = dataset['v_diff'].fillna(0)

    # # Fit and transform the 'category' column
    # dataset['dow'] = pd.to_datetime(dataset['day']).dt.dayofweek
    # dataset['dom'] = pd.to_datetime(dataset['day']).dt.day
    # dataset['year'] = pd.to_datetime(dataset['day']).dt.year

    # # Bottom out year and hour so they fit into their representative embedding space
    # dataset['year'] = dataset['year'] - dataset['year'].min()
    # dataset['hour'] = dataset['hour'] - dataset['hour'].min()
    # dataset['bb_category'] = dataset['bb_category'] + 1

    # Split data into train, val, test (assuming 70% train, 15% val, 15% test)
    train_idx = int(math.floor(0.80 * len(alerts_df)))
    # val_idx = int(0.20 * len(alerts_df))
    train_alerts = alerts_df.iloc[:train_idx]
    val_alerts = alerts_df.iloc[train_idx:]

    # features_symbol = model_hyperparameters['numerical_features'] + model_hyperparameters['categorical_features'] + ['group_id']
    # num_features = model_hyperparameters['numerical_features']
    # cat_features = model_hyperparameters['categorical_features']
    # features_train = dataset[features_symbol]
    # features_train.dropna(inplace=True)
    # features_train = features_train.replace([np.inf, -np.inf], np.nan)

    # tr = features_train.ffill().interpolate()

    # train_data = tr.loc[tr['group_id'].isin(group_ids[:75])]
    # val_data = tr.loc[tr['group_id'].isin(group_ids[75:100])]
    # test_data = tr.loc[tr['group_id'].isin(group_ids[150:200])]

    X_train, X_train_cat, target_train = create_trend_sequences(ts_data, int(model_hyperparameters['context_length']),int(model_hyperparameters['prediction_horizon']),model_hyperparameters,train_alerts)
    X_val, X_val_cat, target_val = create_trend_sequences(ts_data, int(model_hyperparameters['context_length']),int(model_hyperparameters['prediction_horizon']),model_hyperparameters,val_alerts)
    # X_test, X_test_cat = create_trend_sequences(test_data, model_hyperparameters['context_length'],model_hyperparameters['prediction_horizon'],num_features, cat_features)


    print(f"Numerical train shape: {X_train.shape}")
    print(f"Categorical train shape: {X_train_cat.shape}")
    print(f"Target train shape: {target_train.shape}")

    print(f"Numerical train shape: {X_val.shape}")
    print(f"Categorical train shape: {X_val_cat.shape}")
    print(f"Target train shape: {target_val.shape}")

    # print(X_train[0])
    # Create DataLoaders
    train_data = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.LongTensor(X_train_cat), 
        torch.FloatTensor(target_train), 
    )
    val_data = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.LongTensor(X_val_cat), 
        torch.FloatTensor(target_val), 
    )
    # test_data = TensorDataset(
    #     torch.FloatTensor(X_test), 
    #     torch.LongTensor(X_test_cat), 
    #     # torch.FloatTensor(Y_test), 
    # )

    train_loader = DataLoader(train_data, batch_size=model_hyperparameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=model_hyperparameters['batch_size'])
    # test_loader = DataLoader(test_data, batch_size=model_hyperparameters['batch_size'])


    return train_loader, val_loader

def scale_df(raw_data, model_hyperparameters):
    dataset = raw_data.copy()
    temporal_scalers = {}
    unscaled_list = []
    scaled_list = []

    dataset['ymd'] = dataset.apply(lambda x: x['date'].split(" ")[0], axis=1)
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset['dow'] = dataset['date'].apply(lambda x: x.dayofweek)
    dataset['year'] = dataset['date'].apply(lambda x: x.year)
    dataset['hour'] = dataset['date'].apply(lambda x: x.hour)
    dataset['dom'] = dataset['date'].apply(lambda x: x.day)
    dataset['month'] = dataset['date'].apply(lambda x: x.month)
    dataset['minute'] = dataset['date'].apply(lambda x: x.minute)
    dataset['dt'] = dataset.apply(lambda x: f"{x['ymd']} {x['hour']}:{x['minute']}", axis=1)

    dataset['year'] = dataset['year'] - dataset['year'].min()
    dataset['hour'] = dataset['hour'] - dataset['hour'].min()
    dataset['bb_category'] = dataset['bb_category'] + 1

    ## take the column symbol and create a column symbol_encoded which is a numerical representation
    dataset['symbol_encoded'] = LabelEncoder().fit_transform(dataset['symbol'])
    print(dataset['symbol_encoded'].unique())


    print(model_hyperparameters['categorical_features'])
    ts_cat_features = model_hyperparameters['categorical_features'][:-1]
    print(ts_cat_features)

    features_symbol = model_hyperparameters['numerical_features'] + ['symbol','dt','ymd'] + ts_cat_features
    features_train = dataset[features_symbol]
    cat_ad = ts_cat_features + ['symbol','dt','ymd']

    # unscaled_df = features_train.dropna()
    # features_train =features_train.replace([np.inf, -np.inf], np.nan)
    # tr = features_train.ffill().interpolate()

    for symbol in dataset['symbol'].unique():
        temporal_scalers[symbol] = TemporalFeatureScaler(window_size=(model_hyperparameters['context_length']*model_hyperparameters['data_window_length']), min_periods=10, scaling_method='rolling', prediction_horizon=model_hyperparameters['prediction_horizon']) 

        # Scale train data
        symbol_train = features_train[features_train['symbol'] == symbol]
        symbol_train = symbol_train.replace([np.inf, -np.inf], np.nan)
        symbol_train = symbol_train.ffill().interpolate()

        symbol_train_scaled = temporal_scalers[symbol].fit_transform(symbol_train[model_hyperparameters['numerical_features']])
        for cat_feature in cat_ad:
            if cat_feature == 'upside_threshold_prediction' or cat_feature == 'downside_threshold_prediction':
                continue
            symbol_train_scaled[cat_feature] = symbol_train[cat_feature].values
        
        unscaled_list.append(symbol_train)
        scaled_list.append(symbol_train_scaled)


    # # Reassemble the datasets
    scaled = pd.concat(scaled_list)
    unscaled = pd.concat(unscaled_list)


    del dataset
    return scaled, unscaled, temporal_scalers

def create_random_batches(df, num_batches, batch_size):
    # Shuffle the dataframe
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Calculate total number of rows needed
    total_rows = num_batches * batch_size
    
    # If we don't have enough rows, we'll need to sample with replacement
    if total_rows > len(df):
        df_shuffled = df_shuffled.sample(n=total_rows, replace=True).reset_index(drop=True)
    else:
        # If we have more than enough rows, just take what we need
        df_shuffled = df_shuffled.iloc[:int(total_rows)]
    
    # Split into batches
    batches = np.array_split(df_shuffled, num_batches)
    
    return batches

def create_target_values(mode, alerts, predictions, targets):
        formatted_results = []
        for idx, alert in alerts.iterrows():
            if mode == 'max_estimation':
                prediction_values = predictions[idx].tolist()
                target_value = targets[idx, :, 1].tolist()
                target_max = max(target_value)
                target_idx = target_value.index(target_max)
                prediction_max = max(prediction_values)
                prediction_idx = prediction_values.index(prediction_max)
                alert['target'] = target_max
                alert['prediction'] = prediction_max
                alert['target_index'] = target_idx
                alert['prediction_index'] = prediction_idx
                formatted_results.append(alert)
            elif mode == 'min_estimation':
                prediction_values = predictions[idx]
                target_value = targets[idx, :, 2].tolist()
                target_min = min(target_value)
                target_idx = target_value.index(target_max)
                prediction_min = min(prediction_values)
                prediction_idx = prediction_values.index(prediction_max)
                alert['target'] = target_min
                alert['prediction'] = prediction_min
                alert['target_index'] = target_idx
                alert['prediction_index'] = prediction_idx
                formatted_results.append(alert)

        df = pd.DataFrame(formatted_results)
        return df

# def create_target_values_unscaled(mode, predictions, targets):
#         if mode == 'max_estimation':
#             prediction_values = predictions[:, :, 1]
#             target_values = targets[:, :, 1]
#             print(f"Targets: {target_values.shape}")
#             target_value = targets.max()
#             print(f"Target Value: {target_value}")
#             prediction_value = torch.max(prediction_values, dim=1).values
#         elif mode == 'min_estimation':
#             target_value = torch.max(targets, dim=1).values
#             prediction_value = torch.max(predictions, dim=1).values

#         return prediction_value, target_value

def inverse_scale_predictions(predictions,temporal_scalers,test_alerts,full_df,ts_data,model_hyperparameters):
    predictions_unscaled = []
    print(test_alerts)
    for idx, prediction in enumerate(predictions):
        alert = test_alerts.iloc[idx]
        symbol = alert['symbol']
        symbol_df = full_df[full_df['symbol'] == symbol].reset_index(drop=True)
        target_scaler = temporal_scalers[symbol]
        match_dt = f"{alert['date']} {alert['hour']}:{alert['minute']}"
        try:
            alert_index = symbol_df[symbol_df['dt'] == match_dt].index[0]
        except Exception as e:
            if symbol == 'META':
                print(f"Error in inverse scaling for {symbol} at {match_dt}")
                continue
            else:
                # print(f"Alert: {alert}")
                # print(symbol_df)
                # print(match_dt)
                print(e)
                continue
        if model_hyperparameters['mode'] == 'max_estimation':
            prediction_unscaled = target_scaler.inverse_transform(prediction,'h',alert_index,model_hyperparameters)
        elif model_hyperparameters['mode'] == 'min_estimation':
            prediction_unscaled = target_scaler.inverse_transform(prediction,'l',alert_index,model_hyperparameters)
        predictions_unscaled.append(prediction_unscaled)
    return predictions_unscaled


## Archived class very useful for building technical indicators for fake data
# class TechnicalIndicators(nn.Module):
#     def __init__(self, window_size):
#         super().__init__()
#         self.window_size = window_size

#     def forward(self, prices):
#         # prices shape: (batch_size, sequence_length)
        
#         # Compute Moving Average
#         ma = torch.cumsum(prices, dim=1) / torch.arange(1, prices.shape[1] + 1).unsqueeze(0).to(prices.device)
        
#         # Compute Relative Strength Index (RSI)
#         diff = prices[:, 1:] - prices[:, :-1]
#         gain = torch.maximum(diff, torch.zeros_like(diff))
#         loss = torch.maximum(-diff, torch.zeros_like(diff))
        
#         avg_gain = torch.zeros_like(prices)
#         avg_loss = torch.zeros_like(prices)
        
#         avg_gain[:, self.window_size:] = torch.cumsum(gain, dim=1)[:, self.window_size-1:] / self.window_size
#         avg_loss[:, self.window_size:] = torch.cumsum(loss, dim=1)[:, self.window_size-1:] / self.window_size
        
#         rs = avg_gain / (avg_loss + 1e-8)
#         rsi = 100 - (100 / (1 + rs))

#         return torch.stack([ma, rsi], dim=-1)

# def generate_dummy_data(num_samples, seq_length, num_numerical, num_categorical):
#     # Initialize the data array
#     numerical_data = np.zeros((num_samples, num_numerical))
#     total_samples = num_samples 

    
#     # Generate data for each feature using different functions
#     for i in range(num_numerical):
#         x = np.linspace(0, 10, num_samples)
#         print(x)
#         if i % 5 == 0:
#             numerical_data[:, i] = np.sin(x)
#         elif i % 5 == 1:
#             numerical_data[:, i] = np.cos(x)
#         elif i % 5 == 2:
#             numerical_data[:, i] = np.log(x + 1)
#         elif i % 5 == 3:
#             numerical_data[:, i] = (np.exp(x / 10))*3
#         elif i % 5 == 4:
#             numerical_data[:, i] = (np.sin(x)*3)
    
#     categorical_data = np.random.randint(0, 3, (total_samples, len(num_categorical)))
#     cat_seq = create_sequences(categorical_data, seq_length)
#     num_seq = create_sequences(numerical_data, seq_length)

#     # Calculate the target column as the sum of all feature columns
#     target = numerical_data.sum(axis=1)[50:]
    
#     # Calculate volatility as the standard deviation of the feature columns for each sample
#     volatility = np.std(numerical_data, axis=1)[50:]
    
#     return  num_seq, cat_seq, target, volatility

# def build_dummy_datasets(sequence_length, num_numerical_features, num_categories, batch_size, dataset_size):
#     # Generate dummy data
#     numerical_x, categorical_x, price_movement, volatility = generate_dummy_data(dataset_size, sequence_length, num_numerical_features, num_categories)

#     # Now all outputs should have 10000 samples
#     print(f"Numerical shape: {numerical_x.shape}")
#     print(f"Categorical shape: {categorical_x.shape}")
#     print(f"Price movement shape: {price_movement.shape}")
#     print(f"Volatility shape: {volatility.shape}")

#     # Split the data
#     num_train, num_val, cat_train, cat_val, price_train, price_val, vol_train, vol_val = train_test_split(
#         numerical_x, categorical_x, price_movement, volatility, test_size=0.2, random_state=42
#     )

#     # Create DataLoaders
#     train_data = TensorDataset(
#         torch.FloatTensor(num_train), 
#         torch.LongTensor(cat_train), 
#         torch.FloatTensor(price_train), 
#         torch.FloatTensor(vol_train)
#     )
#     val_data = TensorDataset(
#         torch.FloatTensor(num_val), 
#         torch.LongTensor(cat_val), 
#         torch.FloatTensor(price_val), 
#         torch.FloatTensor(vol_val)
#     )

#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size)

#     return train_loader, val_loader