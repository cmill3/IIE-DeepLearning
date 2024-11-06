import optuna
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import data_helper as data
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pandas as pd

from layers.patchTST_cat import PatchTSTCat
from trainer import Trainer

feature_data = {            
            "numerical_features": [
            'c','v','o','close_diff','h','l'
            ],
            "evaluation_columns": ['c'],
            "categorical_features": ['symbol_encoded','hour','dow','bb_category','year'],
            "categorical_cardinalities": [6,7,5,3,7],
}

# Define your initial configuration
initial_config = {
    "num_epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.001,
    "context_length": 240,
    "attention_heads": 8,
    "hidden_state_dim": 64,
    "feedforward_dim": 64,
    "dropout": 0.2,
    "transformer_layers": 4,
    "patch_len": 16,
    "stride": 0.5,
    "batches_per_epoch": 250,
    "data_window_length": 2,
    "categorical_embedding_dim": 16,
    "categorical": True,
    "prediction_horizon": 8,
    "optimizer": 'adam',
    "decomposition": False,
    "kernel_size": 25,
}


def build_optimizer(model, model_hyperparameters):
    ## check a conditional for optiimizer type between ADAM,rmsprop,sgd
    # if model_hyperparameters['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters["learning_rate"])
    # elif model_hyperparameters['optimizer'] == 'rmsprop':
    #     optimizer = optim.RMSprop(model.parameters(), lr=model_hyperparameters["learning_rate"])
    # elif model_hyperparameters['optimizer'] == 'sgd':
    #     optimizer = optim.SGD(model.parameters(), lr=model_hyperparameters["learning_rate"])
    return optimizer

def objective(trial):
    file_path = "/Users/charlesmiller/Code/PycharmProjects/FFACAP/Icarus/DL_experiments/project_c/models/patchTST/datasets/full.csv"

    # Define the hyperparameters to optimize
    model_hyperparameters = {
        "num_epochs": 100,
        "batch_size": 256,
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "context_length": trial.suggest_categorical("context_length", [120, 240, 300]),
        "attention_heads": trial.suggest_categorical("attention_heads", [4, 8, 16]),
        "hidden_state_dim": trial.suggest_categorical("hidden_state_dim", [64, 96]),
        "feedforward_dim": trial.suggest_categorical("feedforward_dim", [64, 96]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        "transformer_layers": trial.suggest_int("transformer_layers", 2, 6),
        "patch_len": trial.suggest_categorical("patch_len", [8, 16, 32]),
        "stride": trial.suggest_categorical("stride", [0.25, 0.5]),
        "batches_per_epoch": trial.suggest_categorical("batches_per_epoch", [100, 250, 500]),
        "data_window_length": trial.suggest_int("data_window_length", 2, 3),
        "categorical_embedding_dim": trial.suggest_categorical("categorical_embedding_dim", [8, 16, 24]),
        "prediction_horizon": trial.suggest_categorical("prediction_horizon", [4, 8, 16]),
        "decomposition": trial.suggest_categorical("decomposition", [True, False]),
        "kernel_size": 25,
    }

    # Add the feature data
    model_hyperparameters.update(feature_data)

    # Load and preprocess data
    train_loader, val_loader, _, _, _, _, _ = data.build_market_datasets_categorical(
        file_path, model_hyperparameters=model_hyperparameters
    )


    # Initialize the model
    model = PatchTSTCat(model_hyperparameters)

    # Build optimizer
    optimizer = build_optimizer(model, model_hyperparameters)

    # Train the model
    trainer = Trainer()
    trainer.setup(model, optimizer)
    try:
        val_loss = trainer.train_categorical(train_loader, val_loader, model_hyperparameters, [0])
    except Exception as e:
        trainer.cleanup()
        raise optuna.exceptions.TrialPruned()

    trainer.cleanup()

    return val_loss

def save_study_results(study):
    # Save study object
    joblib.dump(study, "optuna_study.pkl")
    
    # Save all trials information to a CSV file
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    
    # Save best trial information
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    with open("best_trial_results.txt", "w") as f:
        f.write(f"Best Value: {best_value}\n")
        f.write("Best Parameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")

def optuna_sweep():
    # Create a new study object with Bayesian optimization as the sampler
    sampler = optuna.samplers.TPESampler(seed=42)  # TPE stands for Tree-structured Parzen Estimator, which is a Bayesian optimization algorithm
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Add the initial configuration as the first trial
    study.enqueue_trial(initial_config)

    # Optimize the study
    study.optimize(objective, n_trials=100)


    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Print study statistics
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]))
    print("  Number of complete trials: ", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))

    # Save study results
    save_study_results(study)


if __name__ == "__main__":
    optuna_sweep()