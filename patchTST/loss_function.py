import pandas as pd
import numpy as np


def custom_loss_function(predictions, targets, tolerance):
    # Ensure inputs are floating point tensors and require gradients
    predictions = predictions.float()
    targets = targets.float()

    # Calculate the relative difference
    diff = (predictions - targets) / (targets + 1e-8)  # Add small epsilon to avoid division by zero

    tol_diff = 1 - tolerance
    upper_bound = tol_diff * targets
    lower_bound = tolerance * targets
    mask = (diff >= lower_bound) & (diff <= upper_bound)
    last_values = mask[:, -1, 0]
    bool_values = last_values.bool()
    
    # Calculate accuracy as a tensor
    accuracy = bool_values.float().mean()
    accuracy.requires_grad = True

    # Return 1 - accuracy as the loss
    return 1 - accuracy


    