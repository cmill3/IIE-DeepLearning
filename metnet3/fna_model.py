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
import wandb
from functools import partial
from finNetAtt import FinNetAtt

# Example usage
# num_numerical_features = 5
# num_categories = [10, 5, 3]  # Example: 3 categorical variables with 10, 5, and 3 categories respectively
    
def predict_test_data(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_targets = []
    all_symbols = []

    
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (num_x, cat_x, price_y) in enumerate(tqdm(test_loader, desc="Testing")):
            num_x, cat_x, target = num_x.to(device), cat_x.to(device), price_y.to(device)
            
            # Generate lead times (assuming you want to predict for all lead times)
            symbol = cat_x[:, 0, 0]  # Assuming the symbol is the first categorical feature

            
            # Make predictions
            price_preds = model(num_x, cat_x, symbol)
            
            # Store predictions and targets
            all_predictions.append(price_preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_symbols.append(symbol.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_symbols = np.concatenate(all_symbols, axis=0)
    
    return all_predictions, all_targets, all_symbols

def evaluate_predictions(predictions, targets, symbol_encoded, target_scalers, label_encoder):
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)

    symbols = label_encoder.inverse_transform(symbol_encoded)
    print(symbols)
    print(target_scalers)
    
    # Inverse transform using symbol-specific scalers
    predictions_original = np.zeros_like(predictions).reshape(-1, 1)
    print(predictions_original)
    targets_original = np.zeros_like(targets).reshape(-1, 1)
    print(targets_original)
    for symbol in np.unique(symbols):
        mask = symbols == symbol
        predictions_original[mask] = target_scalers[symbol].inverse_transform(predictions[mask].reshape(-1, 1))
        targets_original[mask] = target_scalers[symbol].inverse_transform(targets[mask].reshape(-1, 1))
    mse_original = mean_squared_error(targets_original, predictions_original)
    mae_original = mean_absolute_error(targets_original, predictions_original)
    rmse_original = np.sqrt(mse_original)
    
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    
    return mse, rmse, mae, mse_original, rmse_original, mae_original, predictions_original, targets_original


def model_traing(train_loader, val_loader, model):
    # Define loss functions and optimizer
    price_criterion = nn.HuberLoss()
    volatility_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_hyperparameters["leraning_rate"])

    ## Enable anomaly detection
    autograd.set_detect_anomaly(True)

    device = torch.device("mps")
    model.to(device)

    best_val_loss = float('inf')
    patience = model_hyperparameters["patience"]
    patience_counter = 0

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    for epoch in range(model_hyperparameters["num_epochs"]):
        model.train()
        train_loss = 0.0
        

        for batch_idx, (num_x, cat_x, price_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}")):
            if batch_idx >= model_hyperparameters["batches_per_epoch"]:
                break
            try:
                num_x, cat_x, price_y = num_x.to(device), cat_x.to(device), price_y.to(device)
            
                
                optimizer.zero_grad()
                symbol = cat_x[:, 0, 0]  # Assuming the symbol is the first categorical feature        
                price_pred = model(num_x, cat_x, symbol)
                # Convert price_pred to numpy array
                # price_pred_np = price_pred.squeeze().detach().cpu().numpy().reshape(-1, 1)

                # # Inverse transform
                # pred_rescaled = target_scaler.inverse_transform(price_pred_np)

                # # Convert pred_rescaled back to a PyTorch tensor and move to the correct device
                # pred_rescaled_tensor = torch.tensor(pred_rescaled).float().to(price_y.device)

                # print(price_pred.squeeze())
                # print(pred_rescaled.squeeze())
                # print(price_y)

                # Ensure both inputs are PyTorch tensors
                price_loss = price_criterion(price_pred.squeeze(), price_y)
                # vol_loss = volatility_criterion(vol_pred.squeeze(), vol_y)
                
                loss = price_loss
                
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
            except RuntimeError as e:
                print(f"Error in Epoch {epoch+1}, Batch {batch_idx+1}")
                print(f"num_x shape: {num_x.shape}")
                print(f"cat_x shape: {cat_x.shape}")
                print(f"price_y shape: {price_y.shape}")
                # print(f"vol_y shape: {vol_y.shape}")
                # print(f"lead_times shape: {lead_times.shape}")
                print(f"price_pred shape: {price_pred.shape}")
                # print(f"vol_pred shape: {vol_pred.shape}")
                raise e
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        train_loss /= len(train_loader)
                
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for num_x, cat_x, price_y in val_loader:
                num_x, cat_x, price_y = num_x.to(device), cat_x.to(device), price_y.to(device)
                
                symbol = cat_x[:, 0, 0]  # Assuming the symbol is the first categorical feature
                price_pred = model(num_x, cat_x, symbol)

                price_loss = price_criterion(price_pred.squeeze(), price_y)
                # vol_loss = volatility_criterion(vol_pred.squeeze(), vol_y)
                
                val_loss += (price_loss).item()
        
        val_loss /= len(val_loader)
        # Step the scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{model_hyperparameters['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            print("Saving model")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    print("Training completed")
    return device

def model_runner(train_loader, val_loader, test_loader, model_hyperparameters, target_scalers, label_encoder):
    
    # Initialize the model
    device = torch.device("mps")
    model = FinNetAtt(
        num_numerical_features=len(model_hyperparameters["numerical_features"]),
        categorical_cardinalities=model_hyperparameters["categorical_cardinalities"],
        embed_dim=(model_hyperparameters["attention_heads"] * model_hyperparameters["attention_head_dimension"]),
        num_attention_layers=model_hyperparameters["num_attention_layers"],
        attention_heads=model_hyperparameters["attention_heads"],
        attention_head_dimension=model_hyperparameters["attention_head_dimension"],
        dropout=model_hyperparameters["dropout"],
        norm_momentum=model_hyperparameters["norm_momentum"],
        use_sym_embedding=model_hyperparameters["use_sym_embedding"]
    )
    model.to(device)

    device = model_traing(train_loader, val_loader, model)
    # Load the best model weights
    model.load_state_dict(torch.load('best_model.pth'))
    # Make predictions on test data
    predictions, targets, symbols = predict_test_data(model, test_loader, device)
    print(symbols)

    # Evaluate predictions
    mse, rmse, mae, mse_original, rmse_original, mae_original, predictions_original, targets_original = evaluate_predictions(predictions, targets, symbols, target_scalers, label_encoder)

    # # If you need to inverse transform the predictions and targets
    # predictions_original = target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
    # targets_original = target_scaler.inverse_transform(targets.reshape(-1, 1)).reshape(targets.shape)

    # # Evaluate in original scale
    # mse_original, rmse_original, mae_original = evaluate_predictions(predictions_original, targets_original)

    print("Original scale metrics:")
    print(f"MSE: {mse_original:.4f}")
    print(f"RMSE: {rmse_original:.4f}")
    print(f"MAE: {mae_original:.4f}")

    print("Predictions and targets in original scale:")
    print(predictions_original)
    print(targets_original)

    return predictions_original, targets_original



# Hyperparameters
model_hyperparameters = {
"num_epochs": 50,
"batch_size": 256,
"leraning_rate": 0.0001,
"context_length" : 128,
"numerical_features": ['period_volatility','close_diff','bb_spread', 
                       'bb_trend','close_diff24','pct_16T_high', 'pct_16T_low',
                       'stddev_close_24_diff', 'stddev_close_96_diff',
                       'pct_96T_high', 'pct_96T_low','rsi4H','rsi24H', 'macd',
                       'sma_24_trend', 'sma_96_trend','c','h','l','o','v'],
"categorical_features": ['symbol_encoded','hour','dow','bb_category','dom'],
"categorical_cardinalities": [5,10, 5, 3, 31], ## len(list) categorical variables with len(list[i]) categories
"prediction_horizon": 24,
"attention_heads": 2,
"attention_head_dimension": 32,
"dropout": 0.5,
"num_attention_layers": 2,
"patience": 10,
"target": 'c',
"batches_per_epoch": 900,
"norm_momentum": 0.1,
"use_sym_embedding": True
}


if __name__ == "__main__":
    # Generate datasets
    file_path = "/Users/charlesmiller/Documents/Code/IIE/Icarus/DL_experiments/project_c/models/metnet3/datasets/full.csv"
    train_loader, val_loader, test_loader, label_encoder, temporal_scalers, target_scalers = data.build_market_datasets(
        file_path, model_hyperparameters["numerical_features"], model_hyperparameters["categorical_features"],
        target=model_hyperparameters['target'],batch_size=model_hyperparameters["batch_size"], context_length=model_hyperparameters["context_length"],
        prediction_horizon=model_hyperparameters["prediction_horizon"]
        )
    model_runner(train_loader, val_loader, test_loader, model_hyperparameters, target_scalers, label_encoder)