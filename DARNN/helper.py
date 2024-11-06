import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings 

warnings.filterwarnings("ignore")

def process_group(id, data, group_id, label, window_size):
    print(f"Processing {id}")
    group_data = data[data[group_id] == id].tail(window_size)
    if len(group_data) < window_size:
        print(f"Skipping {id} due to insufficient data")
        return None
    return group_data

def ts_transformations(data, group_id, label, window_size):
    group_ids = data[group_id].unique()
    X_values = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_group, id, data, group_id, label, window_size) for id in group_ids]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                X_values.append(result)
    windowed_data = pd.concat(X_values)
    return windowed_data

# def prepare_data(data, features_cont, features_cat, window_size, alerts_df):
#     data = add_labels(data, alerts_df)
#     X, y = ts_transformations(data, 'alert_identifier', features_cont, features_cat, 'one_max_vol_label', window_size)
#     return X, y

def prepare_data_with_window(data, window_size, alerts_df):
    data = add_labels(data, alerts_df)
    X = ts_transformations(data, 'alert_identifier', label='one_max_vol_label', window_size=window_size)
    return X

def add_labels(data, alerts_df):
    alerts_df['one_max_vol'] = (alerts_df['one_max'] / alerts_df['return_vol_5D']).round(3)
    alerts_df['three_max_vol'] = (alerts_df['three_max'] / alerts_df['return_vol_5D']).round(3)
    oneD_target = alerts_df['one_max_vol'].quantile(0.6).round(3)
    threeD_target = alerts_df['three_max_vol'].quantile(0.6).round(3)
    alerts_df['one_max_vol_label'] = alerts_df['one_max_vol'].apply(lambda x: 1 if x >= oneD_target else 0)
    alerts_df['three_max_vol_label'] = alerts_df['three_max_vol'].apply(lambda x: 1 if x >= threeD_target else 0)
    data = data.merge(alerts_df[['alert_identifier', 'three_max_vol_label', 'one_max_vol_label']], on='alert_identifier', how='left')
    return data

if __name__ == "__main__":
    data_path = '/Users/charlesmiller/Documents/ts_data/day_aggs/all.csv'
    alerts_path = '/Users/charlesmiller/Documents/model_tester_data/BF/2015-01-01_2024-05-03EXP.csv'
    df = pd.read_csv(data_path)
    df['symbol'] = df['alert_identifier'].apply(lambda x: x.split('-')[0])
    df['alert_identifier'] = df['alert_identifier'].apply(lambda x: x.split('.')[0])
    alerts_df = pd.read_csv(alerts_path)
    alerts_df['alert_identifier'] = alerts_df.apply(lambda row: f"{row['symbol']}-{row['date']}-{row['hour']}", axis=1)
    trimmed = alerts_df.loc[alerts_df['symbol'].isin(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'NFLX', 'NVDA', 'QQQ', 'SPY', 'IWM'])]
    # trimmed = trimmed.sample(45)
    group_ids = trimmed['alert_identifier'].unique().tolist()
    df = df.loc[df['alert_identifier'].isin(group_ids)]
    config = {'window_size': 20}  # Update with your window size
    agg_data = prepare_data_with_window(df, window_size=20, alerts_df=trimmed)
    agg_data.to_csv('data_window20.csv', index=False)