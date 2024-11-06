import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# def ts_transformations(data,group_id,features_cont,features_cat,label,window_size):
#     ## cat features are already encoded so they dont need further processing
#     features = features_cont + features_cat
#     group_ids = data[group_id].unique()
#     X_values, labels = [], []
#     for id in group_ids:
#         scaler = MinMaxScaler()
#         group_data = data[data[group_id] == id].tail(window_size)
#         if len(group_data) < window_size:
#             print(f"Skipping {id} due to insufficient data")
#             continue
#         y = group_data[label].iloc[-1]
#         X = scaler.fit_transform(group_data[features])

#         X_values.append(X)
#         labels.append(y)
    
#     X_tensor = torch.tensor(X_values).float().transpose(1, 2)
#     y_tensor = torch.tensor(labels).long()
#     return X_tensor, y_tensor

def ts_transformations(data,group_id,features_cont,features_cat,label,window_size):
    ## cat features are already encoded so they dont need further processing
    features = features_cont + features_cat
    group_ids = data[group_id].unique()
    X_values, labels = [], []
    for id in group_ids:
        group_data = data[data[group_id] == id].tail(window_size)
        if len(group_data) < window_size:
            print(f"Skipping {id} due to insufficient data")
            continue
        X_values.append(group_data[features].values)

    windowed_data = pd.DataFrame(X_values,columns=features)
    return windowed_data

def prepare_data(data,features_cont,features_cat, window_size,alerts_df):
    data = add_labels(data,alerts_df)
    X, y = ts_transformations(data,'alert_identifier',features_cont,features_cat,'three_max_vol_label',window_size)
    return X, y

def prepare_data_with_window(data,features_cont,features_cat, window_size,alerts_df):
    data = add_labels(data,alerts_df)
    X, y = ts_transformations(data,'alert_identifier',features_cont,features_cat,'three_max_vol_label',window_size)
    return X, y


def add_labels(data,alerts_df):
    alerts_df['one_max_vol'] = (alerts_df['one_max']/alerts_df['return_vol_10D']).round(3)
    alerts_df['three_max_vol'] = (alerts_df['three_max']/alerts_df['return_vol_10D']).round(3)
    oneD_target = alerts_df['one_max_vol'].quantile(0.6).round(3)
    threeD_target = alerts_df['three_max_vol'].quantile(0.6).round(3)
    alerts_df['one_max_vol_label'] = alerts_df['one_max_vol'].apply(lambda x: 1 if x >= oneD_target else 0)
    alerts_df['three_max_vol_label'] = alerts_df['three_max_vol'].apply(lambda x: 1 if x >= threeD_target else 0)
    alerts_df['alert_identifier'] = alerts_df.apply(lambda row: f"{row['symbol']}-{row['date']}-{row['hour']}",axis=1)
    data = data.merge(alerts_df[['alert_identifier','three_max_vol_label','one_max_vol_label']],on='alert_identifier',how='left')
    return data