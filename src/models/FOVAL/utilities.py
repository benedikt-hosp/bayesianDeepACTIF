import random
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader

from src.dataset_classes.TimeSeriesDataset import TimeSeriesDataset

device = torch.device("cuda:0")  # Replace 0 with the device number for your other GPU


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_optimizer(learning_rate, weight_decay, model=None):
    # Set a seed before initializing your model
    # seed_everything(seed=42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=learning_rate)  # 100

    return optimizer, scheduler


# def create_lstm_tensors_dataset(X, y):
#     # Ensure inputs are not None or empty
#     if not (X and y):
#         raise ValueError("X or y is empty or None")
#
#     # Convert training features efficiently
#     # features_tensor = torch.as_tensor(np.stack([seq.values for seq in X]), dtype=torch.float32)
#     features_tensor = torch.as_tensor(np.stack(X), dtype=torch.float32)  # No .values needed
#
#     # Convert training targets
#     targets_tensor = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)
#
#     print(
#         f"Dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} and {len(y)} labels")
#
#     return features_tensor, targets_tensor

def create_lstm_tensors_dataset(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X or y is empty")

    # Convert features and targets to PyTorch tensors efficiently
    features_tensor = torch.tensor(np.array(X), dtype=torch.float32)  # Direct conversion
    targets_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    print(f"Dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} "
          f"and {len(y)} labels")

    return features_tensor, targets_tensor


#     # Check for None or empty inputs
#     assert X is not None and len(X) > 0, "X is empty or None"
#     assert y is not None and len(y) > 0, "y is empty or None"
#
#     # Convert training features
#     features = np.array([sequence.values for sequence in X])
#     print(f"features shape: {features.shape}")
#     features_tensor = torch.tensor(features, dtype=torch.float32)
#
#     # Convert training targets
#     targets_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
#
#     print(
#         f"Dataset has {features_tensor.shape[0]} samples with sequence shape {features_tensor.shape[1:]} and {len(y)} labels")
#
#     return features_tensor, targets_tensor

#
# def create_dataloaders_dataset(features_tensor, targets_tensor, is_train, batch_size):
#     # train_dataset = TensorDataset(features_tensor, targets_tensor)
#     # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=is_train,
#     #                           drop_last=False, pin_memory=True, num_workers=4, prefetch_factor=2)
#
#     dataset = TimeSeriesDataset(features_tensor, targets_tensor)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#
#     return data_loader

def create_dataloaders_dataset(features_tensor, targets_tensor, is_train, batch_size):
    dataset = TimeSeriesDataset(features_tensor, targets_tensor)

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4,
                      pin_memory=True, prefetch_factor=2, persistent_workers=True)


# def analyzeResiduals(predictions, actual_values):
#     # Calculate absolute errors
#     absolute_errors = np.abs(predictions - actual_values)
#
#     # 1. CALCULATE THE AMOUNT OF GREAT OK AND BAD errors
#     """
#         Desc: Show how many good, ok, and bad estimations we have (remember to show distribution too)
#         Categorize errors into bins
#         bin 1: < 1 cm
#         bin 2: < 10 cm
#         bin 3: > 20 cm
#     """
#     bin1 = 1
#     bin2 = 10
#     bin3 = 20
#
#     errors_under_1cm = np.sum(absolute_errors < bin1) / len(absolute_errors)
#     errors_1cm_to_10cm = np.sum((absolute_errors >= bin1) & (absolute_errors <= bin2)) / len(absolute_errors)
#     errors_10cm_to_20cm = np.sum((absolute_errors >= bin2) & (absolute_errors < bin3)) / len(absolute_errors)
#     errors_above_20cm = np.sum(absolute_errors > bin3) / len(absolute_errors)
#
#     # Print percentages
#     print(f"Errors under 1 cm: {errors_under_1cm * 100:.2f}%")
#     print(f"Errors between 1 cm and 10 cm: {errors_1cm_to_10cm * 100:.2f}%")
#     print(f"Errors between 10 cm and 20 cm: {errors_10cm_to_20cm * 100:.2f}%")
#     print(f"Errors above 20 cm: {errors_above_20cm * 100:.2f}%")
#
#     # == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
#     # 2. Calculate which ranges (in bins of 10 cm )where predicted with wich average error
#     """
#         Desc: Show which depth values can be predicted the best
#         Analyze performance across different depths
#         bin size: 10
#     """
#     bin_size = 10
#     # Bin actual values into 10 cm intervals
#     bins = np.arange(30, max(actual_values) + bin_size, bin_size)  # Adjust the range as needed
#     bin_indices = np.digitize(actual_values, bins)
#
#     # Calculate mean absolute error for each bin
#     mean_errors_per_bin = []
#
#     for i in range(1, len(bins)):
#         bin_errors = absolute_errors[bin_indices == i]
#         mean_error = np.nanmean(bin_errors) if len(bin_errors) > 0 else 0.0
#         if mean_error != 0.0:
#             mean_errors_per_bin.append(mean_error)
#
#     # Print mean errors per bin
#     for i, error in enumerate(mean_errors_per_bin):
#         print(f"Depths bin: {bins[i]} to {bins[i + 1]} cm: MAE: {error:.2f} cm")
#
#     # calculate average error between 0.35 and 2 meters
#     lower_bound = 35
#     upper_bound = 200
#
#     # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
#     filtered_errors_for_specific_actual_range = absolute_errors[
#         (actual_values >= lower_bound) & (actual_values <= upper_bound)]
#
#     # Calculate the mean of these filtered errors
#     average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)
#
#     print(
#         f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")
#
#     # calculate average error between 0 and 6 meters
#     lower_bound = 0
#     upper_bound = 600
#
#     # Filter absolute_errors based on the condition that actual_values are between 0.35m and 2m
#     filtered_errors_for_specific_actual_range = absolute_errors[
#         (actual_values >= lower_bound) & (actual_values <= upper_bound)]
#
#     # Calculate the mean of these filtered errors
#     average_error_for_specific_actual_range = np.mean(filtered_errors_for_specific_actual_range)
#
#     print(
#         f"MAE for depth range between {lower_bound} and {upper_bound} cm: {average_error_for_specific_actual_range:.2f} cm")
#
#     results = {
#         '<1cm': errors_under_1cm * 100,
#         '1-10cm': errors_1cm_to_10cm * 100,
#         '10-20': errors_10cm_to_20cm * 100,
#         '>20': errors_above_20cm * 100,
#         'mean_errors_per_bin': {f"{bins[i]} to {bins[i + 1]} cm": error for i, error in
#                                 enumerate(mean_errors_per_bin)},
#         'average_error_for_2m_range': average_error_for_specific_actual_range
#     }
#     return results

def analyzeResiduals(predictions, actual_values):
    absolute_errors = np.abs(predictions - actual_values)

    # Binning error categorization
    bins = [1, 10, 20]
    error_percentages = [np.mean(absolute_errors < bins[0]),
                         np.mean((absolute_errors >= bins[0]) & (absolute_errors <= bins[1])),
                         np.mean((absolute_errors >= bins[1]) & (absolute_errors < bins[2])),
                         np.mean(absolute_errors > bins[2])]

    # Print error percentages
    bin_labels = ["<1 cm", "1-10 cm", "10-20 cm", ">20 cm"]
    for label, perc in zip(bin_labels, error_percentages):
        print(f"Errors {label}: {perc * 100:.2f}%")

    # Bin analysis using NumPy digitization
    bin_size = 10
    bin_edges = np.arange(30, max(actual_values) + bin_size, bin_size)
    bin_indices = np.digitize(actual_values, bin_edges)

    mean_errors_per_bin = [np.mean(absolute_errors[bin_indices == i]) if np.any(bin_indices == i) else 0
                           for i in range(1, len(bin_edges))]

    # Print bin-wise errors
    for i, error in enumerate(mean_errors_per_bin):
        print(f"Depths bin: {bin_edges[i]} to {bin_edges[i+1]} cm: MAE: {error:.2f} cm")

    # Predefined depth range analysis
    ranges = [(35, 200), (0, 600)]
    range_errors = {
        f"{low}-{high} cm": np.mean(absolute_errors[(actual_values >= low) & (actual_values <= high)])
        for low, high in ranges
    }

    for k, v in range_errors.items():
        print(f"MAE for depth range {k}: {v:.2f} cm")

    return {
        'error_distribution': dict(zip(bin_labels, [x * 100 for x in error_percentages])),
        'mean_errors_per_bin': {f"{bin_edges[i]}-{bin_edges[i+1]} cm": err for i, err in enumerate(mean_errors_per_bin)},
        'depth_range_errors': range_errors
    }


# def print_results(iteration, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay,
#                   fc1_dim,
#                   avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2,
#                   best_train_mse, best_train_mae, best_train_smae, avg_val_mse, avg_val_rmse, avg_val_mae,
#                   avg_val_smae, avg_val_r2, best_val_mse, best_val_mae, best_val_smae):
#     """ 4. Print current run average values"""
#     print(f"Run: {iteration}:\n"
#           f" TRAINING: MSE:", avg_train_mse, " \tRMSE:", avg_train_rmse, " \tMAE:", avg_train_mae, " \tHuber: ",
#           avg_train_smae, " \tR-squared:", avg_train_r2,
#           " \tBest train MSE:", best_train_mse, "\tBest train MAE:", best_train_mae, "\t Best train Huber:",
#           best_train_smae,
#           " \nVALIDATION: \tMSE:", avg_val_mse, " \tRMSE:", avg_val_rmse, " \tMAE:", avg_val_mae, " \tHuber: ",
#           avg_val_smae, " \tR-squared:", avg_val_r2,
#           " \tBest val MSE:", best_val_mse, " \tBest Val MAE:", best_val_mae, " \tBest val Huber:", best_val_smae,
#           "\n",
#           " Batch size: ", batch_size, " \tDropout: ", dropoutRate, " \tembed_dim :", embed_dim, " \tfc1_dim :",
#           fc1_dim,
#           " \tLearning rate: ", learning_rate, "\tWeight decay: ", weight_decay, "\tL1: ", l1_lambda)

def print_results(iteration, batch_size, embed_dim, dropoutRate, l1_lambda, learning_rate, weight_decay,
                  fc1_dim, avg_train_mse, avg_train_rmse, avg_train_mae, avg_train_smae, avg_train_r2,
                  best_train_mse, best_train_mae, best_train_smae, avg_val_mse, avg_val_rmse, avg_val_mae,
                  avg_val_smae, avg_val_r2, best_val_mse, best_val_mae, best_val_smae):
    """Print formatted results for the training and validation process."""

    train_results = f"""
    Run: {iteration}
    TRAINING:
        MSE: {avg_train_mse:.4f}  |  RMSE: {avg_train_rmse:.4f}  |  MAE: {avg_train_mae:.4f}  
        Huber: {avg_train_smae:.4f}  |  R-squared: {avg_train_r2:.4f}
        Best Train MSE: {best_train_mse:.4f}  |  Best Train MAE: {best_train_mae:.4f}  |  Best Train Huber: {best_train_smae:.4f}
    """

    val_results = f"""
    VALIDATION:
        MSE: {avg_val_mse:.4f}  |  RMSE: {avg_val_rmse:.4f}  |  MAE: {avg_val_mae:.4f}  
        Huber: {avg_val_smae:.4f}  |  R-squared: {avg_val_r2:.4f}
        Best Val MSE: {best_val_mse:.4f}  |  Best Val MAE: {best_val_mae:.4f}  |  Best Val Huber: {best_val_smae:.4f}
    """

    model_params = f"""
    Model Parameters:
        Batch size: {batch_size}  |  Dropout: {dropoutRate:.2f}  |  Embed Dim: {embed_dim}  |  FC1 Dim: {fc1_dim}
        Learning Rate: {learning_rate:.6f}  |  Weight Decay: {weight_decay:.6f}  |  L1 Lambda: {l1_lambda:.6f}
    """

    print(train_results)
    print(val_results)
    print(model_params)