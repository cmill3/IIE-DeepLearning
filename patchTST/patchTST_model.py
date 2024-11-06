import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import random as rand
import torch.autograd as autograd
import data_helper as data
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from layers.patchTST import PatchTST
from layers.patchTST_cat import PatchTSTCat
from trainer import Trainer
from data_helper import create_random_batches, inverse_scale_predictions, create_target_values, create_trend_sequences_test

def debug_print(tensor, name):
    print(f"Debug {name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
    print(f"Sample values:\n{tensor[0, :5, :5]}\n")


def predict_test_data(model, test_alerts, device, full_df):
    model.eval()  # Set the model to evaluation mode
    loss_function = nn.MSELoss()
    all_predictions = []
    all_targets = []
    unscaled_predictions = []
    unscaled_targets = []

    num_batches = len(test_alerts) / MODEL_HYPERPARAMETERS["batch_size"]
    if num_batches < 1:
        num_batches = 1
    print(f"Number of batches: {num_batches}")
    test_alerts = create_random_batches(test_alerts, num_batches, MODEL_HYPERPARAMETERS["batch_size"])
    print(test_alerts)
    ts_data, unscaled_df, temporal_scalers = data.scale_df(full_df, MODEL_HYPERPARAMETERS)
    full_results = []
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (batch_alerts) in enumerate(tqdm(test_alerts, desc="Testing")):
            X_test, X_cat, Y_test, Y_target, new_alerts_df = create_trend_sequences_test(ts_data, MODEL_HYPERPARAMETERS["context_length"], 
                                                                MODEL_HYPERPARAMETERS["prediction_horizon"], MODEL_HYPERPARAMETERS,
                                                                batch_alerts, unscaled_df)
            
            try:
                num_x = torch.FloatTensor(X_test)
                cat_x = torch.LongTensor(X_cat)
                batch_y = torch.FloatTensor(Y_test)
            except:
                print("X_test")
                continue
            
            num_x, batch_y, cat_x = num_x.to(device), batch_y.to(device), cat_x.to(device)
            predictions = model(num_x, cat_x)
            unscaled_predictions = inverse_scale_predictions(predictions, temporal_scalers, new_alerts_df, unscaled_df, ts_data,MODEL_HYPERPARAMETERS)
            formatted_results = create_target_values(MODEL_HYPERPARAMETERS['mode'], new_alerts_df, unscaled_predictions, Y_target)
            full_results.append(formatted_results)
            
    results_df = pd.concat(full_results)
    predictions_loss = loss_function(torch.tensor(results_df['prediction'].values, dtype=torch.float32), torch.tensor(results_df['target'].values, dtype=torch.float32))
    print(f"Predictions loss: {predictions_loss}")
    return results_df

def build_optimizer(model, model_hyperparameters):
    ## check a conditional for optiimizer type between ADAM,rmsprop,sgd
    if model_hyperparameters['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters["learning_rate"])
    elif model_hyperparameters['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=model_hyperparameters["learning_rate"])
    elif model_hyperparameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=model_hyperparameters["learning_rate"])
    return optimizer

def model_training(train_alerts, val_alerts, model, ts_data, model_hyperparameters):
    optimizer = build_optimizer(model, model_hyperparameters)

    trainer = Trainer()
    trainer.setup(model, optimizer)

    if model_hyperparameters['categorical']:
        trainer.train_categorical(train_alerts, val_alerts, model_hyperparameters, ts_data)
    else:
        trainer.train(train_alerts, val_alerts, model_hyperparameters, ts_data)

    return

def model_runner(ts_data, train_alerts, val_alerts, test_alerts, model_hyperparameters, test_data,prediction_mode=False):
    # autograd.set_detect_anomaly(True)
    # Initialize the model
    if model_hyperparameters['categorical']:
            model = PatchTSTCat(
                model_hyperparameters,
            )
    else:
        model = PatchTST(
            model_hyperparameters,
        )

    if prediction_mode:
        model.load_state_dict(torch.load('best_model.pth'))
        device = torch.device("mps")
        model.to(device)
        results_df = predict_test_data(model, test_alerts, device, test_data)
        return results_df
    else:
        model_training(train_alerts,val_alerts,model,ts_data, model_hyperparameters)
        # Load the best model weights
        model.load_state_dict(torch.load('best_model.pth'))
        device = torch.device("mps")
        model.to(device)
        ## Make predictions on test data
        results_df = predict_test_data(model, test_alerts, device, test_data)
        results_df.to_csv("prediction_results.csv")


    return "predictions_original", "targets_original"


def create_scaled_trend_df(file_path, model_hyperparameters):
    # Create a dictionary to store the scalers
    scaled_df = data.build_scaled_datasets(file_path, model_hyperparameters)
    scaled_df.to_csv("scaled_df.csv")


def divide_alerts(alerts_df):
    # alerts_df = alerts_df.iloc[40000:]
    alerts_df = alerts_df.loc[alerts_df['upside_threshold_prediction'] == 1]
    ## srandomly split df into train and test
    train_alerts = alerts_df.sample(frac=0.8, random_state=42)
    val_alerts = alerts_df.drop(train_alerts.index)
    return train_alerts, val_alerts

# Hyperparameters
MODEL_HYPERPARAMETERS = {
    "num_epochs": 200,
    "batch_size": 256,
    "batches_per_epoch": 300,
    "learning_rate": 0.001,
    "context_length" : 240,
                "numerical_features": ['c', 'h', 'l','v','close_diff','close_diff56','std_volatility_4_42_diff',
                                    'bb_spread', 'bb_trend','sma_14_trend', 'sma_56_trend', 'pct_16T_high', 'pct_96T_high', 
                                    'cycle_z_scores','daily_cycle_strength', 'weekly_cycle_strength','range_volatility_detail_1_anomaly',
                                    'std_volatility_detail_1_anomaly','range_volatility_detail_1_to_4_ratio',
                                        'std_volatility_detail_1_to_4_ratio','vti_trend','symbol_vti_diff','vti_corr_56'],
    "categorical_features": ['hour','dow','year','bb_category',
                            'symbol_encoded',
                            'downside_threshold_prediction',
                            ],
    "categorical_cardinalities": [7,5,10,3,22,2],
    "evaluation_columns": ['c'],
    "prediction_horizon": 42,
    "attention_heads": 4,
    "hidden_state_dim": 64,
    "feedforward_dim": 64,
    "dropout": 0.1,
    "transformer_layers": 2,
    "individual": True,
    "patch_len": 16,
    "stride": 4,
    "padding_patch": 'end',
    "revin": False,
    "affine": False,
    "patience": 25,
    "kernel_size": 25,
    "subtract_last": 0,
    "scaling_method": "rolling",
    "optimizer": "adam",
    "categorical": True,
    "decomposition": False,
    "data_window_length": 2,
    "categorical_embedding_dim": 16,
    "mode": "max_estimation",
    "partial_epochs": 10,
    "partial_lr": 0.001,
    "partial_batches_per_epoch": 20,
    "partial_batch_size": 128,
}

if __name__ == "__main__":
    # Generate datasets
    test_data = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/full.csv")
    scaled_data = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/scaled_df.csv")
    alerts_df = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/alerts.csv")
    print("FEATURES")
    print(MODEL_HYPERPARAMETERS["numerical_features"])
    print()

    # create_scaled_trend_df
    scale_df, _, _ = data.scale_df(test_data, MODEL_HYPERPARAMETERS)
    scale_df.to_csv("scaled_df.csv")
    
    # train_alerts, val_alerts, test_alerts = divide_alerts(alerts_df)
    # predictions, targets, = model_runner(
    #     scaled_data, train_alerts, val_alerts, test_alerts ,MODEL_HYPERPARAMETERS, 
    #     test_data
    #     )


# 'zeros': Initializes positional encodings with zeros.
# 'normal' or 'gauss': Initializes with values from a normal distribution.
# 'uniform': Initializes with values from a uniform distribution.
# 'lin1d': Linear 1D positional encoding.
# 'exp1d': Exponential 1D positional encoding.
# 'lin2d': Linear 2D positional encoding.
# 'exp2d': Exponential 2D positional encoding.
# 'sincos': Sinusoidal positional encoding (similar to the original Transformer paper).