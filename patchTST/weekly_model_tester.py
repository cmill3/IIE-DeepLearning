from data_helper import create_random_batches, inverse_scale_predictions, create_target_values, create_trend_sequences_test
from patchTST_model import model_runner, build_optimizer, predict_test_data
from weekly_params import MODEL_HYPERPARAMETERS_2H, MODEL_HYPERPARAMETERS_3D
from layers.patchTST_cat import PatchTSTCat
from trainer import Trainer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import boto3

s3 = boto3.client('s3')



def test_controller(test_data, scaled_data, alerts_df, training_dates, time_period):
    ## turn a dict into a dataframe
    config_df = pd.DataFrame.from_dict(MODEL_HYPERPARAMETERS, orient='index')
    s3.put_object(Bucket='icarus-research-data', Key=f'PTST_weekly_predictions/{TEST_NAME}/model_config.csv', Body=config_df.to_csv())
    for index, date in enumerate(training_dates):
        train_alerts, week_alerts, new_train_data_alerts = divide_alerts_weekly(date, alerts_df, index)
        if index == 0:
            model = training_process(train_alerts, new_week_alerts=None,ts_data=scaled_data, full_train=True, time_period=time_period)
        #     # model.load_state_dict(torch.load('best_model_weekly_update_3D.pth'))
        # else:
        #     # model.load_state_dict(torch.load('best_model_weekly_update_3D.pth'))
        # #     model = training_process(train_alerts=None,new_week_alerts=new_train_data_alerts, ts_data=scaled_data, full_train=False)
        success_str = predict_week(week_alerts, test_data, date["deployment_date"], time_period)
        print(success_str)

def training_process(train_alerts, new_week_alerts, ts_data, full_train, time_period):
    model = PatchTSTCat(MODEL_HYPERPARAMETERS)
    optimizer = build_optimizer(model, MODEL_HYPERPARAMETERS)
    if full_train:
        trainer = Trainer()
        trainer.setup(model, optimizer)
        trainer.train_categorical_weekly(train_alerts, MODEL_HYPERPARAMETERS, ts_data, full_train, time_period)
    else:
        trainer = Trainer()
        model.load_state_dict(torch.load(f'best_model_weekly_start_{time_period}.pth'))
        device = torch.device("mps")
        model.to(device)
        trainer.setup(model, optimizer)
        partial_hyperparameters = MODEL_HYPERPARAMETERS
        partial_hyperparameters['num_epochs'] = MODEL_HYPERPARAMETERS['partial_epochs']
        partial_hyperparameters['learning_rate'] = MODEL_HYPERPARAMETERS['partial_lr']
        partial_hyperparameters['batch_size'] = MODEL_HYPERPARAMETERS['partial_batch_size']
        partial_hyperparameters['batches_per_epoch'] = MODEL_HYPERPARAMETERS['partial_batches_per_epoch']
        trainer.train_categorical_weekly(new_week_alerts, partial_hyperparameters, ts_data, full_train)
    return model

def predict_week(week_alerts, ts_data,dt_str, time_period):
    model = PatchTSTCat(MODEL_HYPERPARAMETERS)
    model.load_state_dict(torch.load(f'best_model_weekly_start_{time_period}.pth'))
    device = torch.device("mps")
    model.to(device)
    ## Make predictions on test_data
    results_df = predict_test_data(model, week_alerts, device, ts_data)
    ## store weekly results
    res = s3.put_object(Bucket='icarus-research-data', Key=f'PTST_weekly_predictions/{TEST_NAME}/{dt_str}.csv', Body=results_df.to_csv())
    return f"finished prediction for {dt_str}"


def create_training_dates_dict(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    dates_list = []
    while start_date <= end_date:
        date_object = {
            "deployment_date": start_date,
            "week_end": start_date + timedelta(days=6),
            "training_data_cutoff": start_date - timedelta(days=2),
            "previous_training_data_cutoff": start_date - timedelta(days=50),
        }
        dates_list.append(date_object)
        start_date += timedelta(days=7)
    return dates_list

def divide_alerts_weekly(date_obj,alerts_df, train_index):
    # alerts_df = alerts_df.iloc[40000:]
    alerts_df = alerts_df.loc[alerts_df['upside_threshold_prediction'] == 1]
    if train_index == 0:
        train_alerts = alerts_df[alerts_df["dt"] <= date_obj["training_data_cutoff"]]
        week_alerts = alerts_df[(alerts_df["dt"] >= date_obj["deployment_date"]) & (alerts_df["dt"] < date_obj["week_end"])]
        return train_alerts, week_alerts, None
    else:
        train_alerts = alerts_df[(alerts_df["dt"] <= date_obj["previous_training_data_cutoff"])]
        new_train_data_alerts = alerts_df[(alerts_df["dt"] < date_obj["deployment_date"]) & (alerts_df["dt"] >= date_obj["previous_training_data_cutoff"])]
        week_alerts = alerts_df[(alerts_df["dt"] >= date_obj["deployment_date"]) & (alerts_df["dt"] < date_obj["week_end"])]
    return train_alerts, week_alerts, new_train_data_alerts

if __name__ == "__main__":
    # Generate datasets
    test_data = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/full.csv")
    scaled_data = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/scaled_df.csv")
    alerts_df = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/alerts.csv")
    alerts_df["dt"] = pd.to_datetime(alerts_df["date"])

    # print(len(alerts_df['symbol'].unique()))
    
    # Create training dates
    ## Has to start on a monday
    training_dates = create_training_dates_dict("2023-10-02", "2024-07-27")
    time_period = "3D"
    TEST_NAME = f"2024testvalsym_{time_period}_241101"
    if time_period == "2H":
        MODEL_HYPERPARAMETERS = MODEL_HYPERPARAMETERS_2H
    else:
        MODEL_HYPERPARAMETERS = MODEL_HYPERPARAMETERS_3D
    test_controller(test_data, scaled_data, alerts_df, training_dates,time_period)