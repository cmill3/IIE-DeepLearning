import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import data_helper as data
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
import time
import psutil
import pandas as pd

from layers.patchTST import PatchTST
from layers.patchTST_cat import PatchTSTCat
from trainer import Trainer

evaluation_columns = [0]

def debug_print(tensor, name):
    print(f"Debug {name}:")
    print(f"Shape: {tensor.shape}")
    print(f"Type: {tensor.dtype}")
    print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
    print(f"Sample values:\n{tensor[0, :5, :5]}\n")

def print_memory_usage():
    print(f"RAM memory % used: {psutil.virtual_memory().percent}")
    print(f"MPS memory used: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")

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
    try:
        if model_hyperparameters['categorical']:
            trainer.train_categorical(train_alerts, val_alerts, model_hyperparameters, ts_data)
        else:
            trainer.train(train_alerts, val_alerts, model_hyperparameters, ts_data)
    except Exception as e:
        print(e)
        trainer.cleanup()
        return
    trainer.cleanup()
    return

def model_runner(model_hyperparameters, ts_data, train_alerts, val_alerts):
    model_hyperparameters['numerical_features'] = feature_data['numerical_features']
    model_hyperparameters['categorical_features'] = feature_data['categorical_features']
    model_hyperparameters['categorical_cardinalities'] = feature_data['categorical_cardinalities']
    model_hyperparameters['scaling_method'] = 'rolling'

    
    with wandb.init(config=model_hyperparameters):
        config = wandb.config
        # Initialize the model
        if model_hyperparameters['categorical']:
            model = PatchTSTCat(
                model_hyperparameters,
            )
        else:
            model = PatchTST(
                model_hyperparameters,
            )

        model_training(train_alerts,val_alerts,model,ts_data, model_hyperparameters)

        time.sleep(60)

        return "predictions_original", "targets_original"

    
def divide_alerts(alerts_df):
    # alerts_df = alerts_df.iloc[40000:]
    alerts_df = alerts_df.loc[alerts_df['upside_threshold_prediction'] == 1]
    ## srandomly split df into train and test
    train_alerts = alerts_df.sample(frac=0.8, random_state=42)
    val_alerts = alerts_df.drop(train_alerts.index)
    return train_alerts, val_alerts

feature_data = {            
            "numerical_features": ['c', 'h', 'l','v','close_diff','close_diff56','std_volatility_4_42_diff',
                                   'bb_spread', 'bb_trend','sma_14_trend', 'sma_56_trend', 'pct_16T_high', 'pct_96T_high', 
                                   'cycle_z_scores','daily_cycle_strength', 'weekly_cycle_strength','range_volatility_detail_1_anomaly',
                                   'std_volatility_detail_1_anomaly','range_volatility_detail_1_to_4_ratio',
                                    'std_volatility_detail_1_to_4_ratio','vti_trend','symbol_vti_diff','vti_corr_56'],
            "categorical_features": ['hour','dow','year','bb_category','symbol_encoded','downside_threshold_prediction'],
            "categorical_cardinalities": [7,5,10,3,22,2],
}


def wandb_sweep():
# Hyperparameters
    sweep_config = {
        'method': 'bayes', 
        'bayes': {
            'bayes_initial_samples': 10,
            'exploration_factor': 0.2,
        },
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        "parameters": {
            "num_epochs": {'values': [100]},
            "batch_size": {'values': [256]},
            "learning_rate": {'values': [0.01, 0.001]},
            "context_length": {'values': [100,140]},
            "attention_heads": {'values': [4,8]},
            "hidden_state_dim": {'values': [32,64]},
            "feedforward_dim": {'values': [32,64]},
            "dropout": {'values': [.05,0.1]},
            "transformer_layers": {'values': [2,4]},
            "patch_len": {'values': [8, 16, 32]},
            "stride": {'values': [.25, .5]},
            "batches_per_epoch": {'values': [300]},
            "data_window_length": {'values': [1,2,3]},
            "categorical_embedding_dim": {'values': [16,32,64]},
            "decomposition": {'values': [True]},
            "categorical": {'values': [True]},
            "prediction_horizon": {'values': [42]},
            "optimizer": {'values': ['adam']},
            "kernel_size": {'values': [25]},
            "mode": {'values': ['max_estimation']},
        }
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="patchTST_trend")

    ts_data = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/scaled_df.csv")
    alerts_df = pd.read_csv("/Users/charlesmiller/Documents/Code/DL_experiments/project_c/models/patchTST/datasets/alerts.csv")
    train_alerts, val_alerts = divide_alerts(alerts_df)

    # Define the objective function for the sweep
    def sweep_train():

        wandb.init(project="patchTST_trend")
        model_runner(wandb.config, ts_data, train_alerts, val_alerts,)

    # Start the sweep
    wandb.agent(sweep_id, function=sweep_train, count=45)  # Run 10 trials

if __name__ == "__main__":
    wandb_sweep()


# 'zeros': Initializes positional encodings with zeros.
# 'normal' or 'gauss': Initializes with values from a normal distribution.
# 'uniform': Initializes with values from a uniform distribution.
# 'lin1d': Linear 1D positional encoding.
# 'exp1d': Exponential 1D positional encoding.
# 'lin2d': Linear 2D positional encoding.
# 'exp2d': Exponential 2D positional encoding.
# 'sincos': Sinusoidal positional encoding (similar to the original Transformer paper).