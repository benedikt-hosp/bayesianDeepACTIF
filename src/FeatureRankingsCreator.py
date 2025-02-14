import os
import timeit
import logging

import numpy as np
import pandas as pd
from memory_profiler import memory_usage
import torch

from src.A1_OLD_AutoML_DeepACTIF import A1_AutoDeepACTIF_AnalyzerBackprop
from src.A2_DeepACTIFAnalyzerForward_old import A2_DeepACTIF_Forward_OLD
from src.A3_AutoML_DeepACTIF import A3_AutoML_DeepACTIF
from src.A4_DeepACTIFAggregatorV1 import A4_DeepACTIF_SingleLayer
from src.A5_AutoML_BayesInference import A5_BayesianInference
from src.A6_AutoML_BayesInference_Optim import A6_BayesianInference_Optim
from src.models.FOVAL.foval_preprocessor import input_features
import json
from src.models.FOVAL.foval import Foval

torch.backends.cudnn.enabled = False
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


class PyTorchModelWrapper_SHAP:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, x):
        # Convert NumPy array to PyTorch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)

        # # Check if the input is 2D or 3D
        # if x_tensor.dim() == 2:
        #     # Reshape it to 3D if it's 2D, assuming shape (batch_size, features) to (batch_size, 1, features)
        #     x_tensor = x_tensor.unsqueeze(1)  # Add a time dimension

        # Forward pass through the model (expects 3D input)
        with torch.no_grad():
            model_output = self.model(x_tensor)  # Model output should still be 3D

        # Aggregate the output over time steps (e.g., mean aggregation)
        if model_output.dim() == 3:
            aggregated_output = model_output.mean(dim=1)  # Aggregate over time (batch_size, features)
            print("Model output is 3D")
        else:
            # If model_output is not 3D, we return it as-is (or handle another case)
            aggregated_output = model_output
            print("Model output is 2D")
            print("Shape: ", aggregated_output.shape)

        # Return the aggregated output as a NumPy array
        return aggregated_output.cpu().numpy()


class FeatureRankingsCreator:
    def __init__(self, modelName, datasetName, dataset, trainer, paths, device):
        self.analyzer = None
        self.modelName = modelName
        self.paths = paths
        self.device = device
        self.trainer = trainer
        self.currentModel = self.trainer.model
        self.currentModelName = modelName
        self.currentDatasetName = datasetName
        self.methods = None
        self.memory_data = None
        self.timing_data = None
        self.selected_features = [value for value in input_features if value not in ('SubjectID', 'Gt_Depth')]
        self.currentDataset = dataset
        self.subject_list = dataset.subject_list
        self.de = None
        self.hyperparameters = None
        self.setup_directories()
        self.feature_importance_results = {}  # Dictionary to store feature importances for each method
        self.timing_data = []
        self.memory_data = []
        self.bayes_analyzer = None
        self.methods = [
            # '1_DeepACTIF_FULL_OLD', #error
            # '2_DeepACTIF_Forward_OLD',
            '3_AutoML_DeepACTIF',     # error
            # '4_DeepACTIF_AggregatorV1',
            # '5_BayesianInference',
            # '6_BayesianInference_Optim',

        ]

    def setup_directories(self):
        os.makedirs(self.paths["results_folder_path"], exist_ok=True)

    def process_methods(self):
        for method in self.methods:
            print(f"Evaluating method: {method}")
            aggregated_importances = self.calculate_ranked_list_by_method(method=method)
            self.sort_importances_based_on_attribution(aggregated_importances, method=method)

    def calculate_ranked_list_by_method(self, method='captum_intGrad'):
        aggregated_importances = []
        all_execution_times = []
        all_memory_usages = []

        if method is "1_DeepACTIF_FULL_OLD":
            self.analyzer = A1_AutoDeepACTIF_AnalyzerBackprop(features=self.selected_features,
                                                              model=self.trainer.model,
                                                              device=self.device,
                                                              use_gaussian_spread=True)

        elif method == '2_DeepACTIF_Forward_OLD':
            self.analyzer = A2_DeepACTIF_Forward_OLD(model=self.trainer.model, features=self.selected_features)
        elif method == '3_AutoML_DeepACTIF':
            self.analyzer = A3_AutoML_DeepACTIF(model=self.trainer.model,
                                                features=self.selected_features,
                                                device=self.device,
                                                use_gaussian_spread=True)
        elif method == '4_DeepACTIF_AggregatorV1':
            self.analyzer = A4_DeepACTIF_SingleLayer(model=self.trainer.model,
                                                     selected_features=self.selected_features, device=self.device)
        elif method == '5_BayesianInference':
            self.analyzer = A5_BayesianInference(model=self.trainer.model,
                                                 features=self.selected_features,
                                                 device=self.device,
                                                 use_gaussian_spread=True)
        elif method == '6_BayesianInference_Optim':
            initial_dataloader, _, input_size = self.getInitialDataLoader(self.subject_list)

            # Erzeuge (oder hole) die Singleton-Instanz:
            self.analyzer = A6_BayesianInference_Optim(self.trainer.model,
                                                       features=self.selected_features,
                                                       device=self.device,
                                                       use_gaussian_spread=True,
                                                       initial_dataloader=initial_dataloader)
        else:
            print("Method unknwon")

            # Angenommen, trainer.model und selected_features sind bereits definiert.
            # self.live_vis = LiveVisualizer(self.selected_features, update_interval=10)
            # self.bayes_analyzer.live_visualizer = self.live_vis

        for i, test_subject in enumerate(self.subject_list):
            # if i > 2:
            #     break

            print(f"Processing subject {i + 1}/{len(self.subject_list)}: {test_subject}")

            # create data loaders
            train_loader, valid_loader, input_size = self.getDataLoaders(test_subject)

            method_func = self.get_method_function(method, valid_loader)
            if method_func:
                execution_time, mem_usage, subject_importances = self.calculate_memory_and_execution_time(method_func)
                if subject_importances is not None and len(subject_importances) > 0:
                    all_execution_times.append(execution_time)
                    all_memory_usages.append(max(mem_usage))
                    aggregated_importances.extend(subject_importances)

        self.save_timing_data(method, all_execution_times, all_memory_usages)
        df_importances = pd.DataFrame(aggregated_importances)
        self.feature_importance_results[method] = df_importances
        return df_importances

    def getDataLoaders(self, test_subject):
        batch_size = 460
        validation_subjects = test_subject
        remaining_subjects = np.setdiff1d(self.subject_list, validation_subjects)

        logging.info(f"Validation subject(s): {validation_subjects}")
        logging.info(f"Training subjects: {remaining_subjects}")

        train_loader, valid_loader, input_size = self.currentDataset.get_data_loader(
            remaining_subjects.tolist(), validation_subjects, None, batch_size=batch_size)

        return train_loader, valid_loader, input_size

    def getInitialDataLoader(self, subjects):
        batch_size = 460

        logging.info(f"All subject(s): {subjects}")

        train_loader, valid_loader, input_size = self.currentDataset.get_data_loader(
            subjects.tolist(), None, None, batch_size=batch_size)

        return train_loader, valid_loader, input_size

    def get_method_function(self, method, valid_loader, load_model=False):
        method_functions = {

            # DeepACTIF V1: Methods (v1, v2, v3)
            'deepactif_input_MEAN': lambda: self.compute_deepactif(valid_loader, hook_location='input',
                                                                   actif_variant='MEAN'),
            'deepactif_input_MEANSTD': lambda: self.compute_deepactif(valid_loader, hook_location='input',
                                                                      actif_variant='MEANSTD'),
            'deepactif_input_INV': lambda: self.compute_deepactif(valid_loader, hook_location='input',
                                                                  actif_variant='INV'),
            'deepactif_input_PEN': lambda: self.compute_deepactif(valid_loader, hook_location='input',
                                                                  actif_variant='PEN'),

            'deepactif_lstm_MEAN': lambda: self.compute_deepactif(valid_loader, hook_location='lstm',
                                                                  actif_variant='MEAN'),
            'deepactif_lstm_MEANSTD': lambda: self.compute_deepactif(valid_loader, hook_location='lstm',
                                                                     actif_variant='MEANSTD'),
            'deepactif_lstm_INV': lambda: self.compute_deepactif(valid_loader, hook_location='lstm',
                                                                 actif_variant='INV'),
            'deepactif_lstm_PEN': lambda: self.compute_deepactif(valid_loader, hook_location='lstm',
                                                                 actif_variant='PEN'),

            'deepactif_penultimate_MEAN': lambda: self.compute_deepactif(valid_loader, hook_location='penultimate',
                                                                         actif_variant='MEAN'),
            'deepactif_penultimate_MEANSTD': lambda: self.compute_deepactif(valid_loader, hook_location='penultimate',
                                                                            actif_variant='MEANSTD'),
            'deepactif_penultimate_INV': lambda: self.compute_deepactif(valid_loader, hook_location='penultimate',
                                                                        actif_variant='INV'),
            'deepactif_penultimate_PEN': lambda: self.compute_deepactif(valid_loader, hook_location='penultimate',
                                                                        actif_variant='PEN'),

            'deepactif_ng': lambda: self.compute_deepactif_ng(valid_loader, actif_variant='INV'),
            'autoDeepactifFull': lambda: self.compute_autodeepactif_full(valid_loader),
            'bayesDeepactif': lambda: self.compute_bayesACTIF(valid_loader),

            '1_DeepACTIF_FULL_OLD': lambda: self.compute_deepactif_(valid_loader, method=method),
            '2_DeepACTIF_Forward_OLD': lambda: self.compute_deepactif_(valid_loader, method=method),
            '3_AutoML_DeepACTIF': lambda: self.compute_deepactif_(valid_loader, method=method),
            '4_DeepACTIF_AggregatorV1': lambda: self.compute_deepactif_(valid_loader, method=method),
            '5_BayesianInference': lambda: self.compute_deepactif_(valid_loader, method=method),
            '6_BayesianInference_Optim': lambda: self.compute_deepactif_(valid_loader, method=method),

        }

        return method_functions.get(method, None)

    '''
    =======================================================================================
    # ACTIF Variants
    =======================================================================================
    '''

    def actif_calculation(self, calculation_function, valid_loader):
        all_importances = []
        total_sum = None
        total_count = 0

        # Disable gradient calculation to save memory during inference
        with torch.no_grad():
            # Loop through the validation loader batch by batch
            for batch in valid_loader:
                inputs, _ = batch  # Assuming batch returns (inputs, labels)
                inputs = inputs.to(self.device)  # Move inputs to the appropriate device

                # Convert inputs to numpy if needed, and compute mean activations over time
                inputs_np = inputs.cpu().numpy()  # Convert inputs to numpy array

                # Compute mean activations over time (axis=1) for each feature
                mean_activation = np.mean(np.abs(inputs_np), axis=1)  # Mean over time steps

                # Call the calculation function (e.g., actif_mean) with the mean activations
                importance = calculation_function(mean_activation)
                all_importances.append(importance)

                # Clear GPU cache after each batch to prevent memory buildup
                del inputs
                torch.cuda.empty_cache()

        # Aggregate all importances
        if all_importances:
            all_mean_importances = np.mean(np.array(all_importances), axis=0)
            sorted_indices = np.argsort(-all_mean_importances)
            sorted_features = np.array(self.selected_features)[sorted_indices]
            sorted_all_mean_importances = all_mean_importances[sorted_indices]

            # Return sorted results as a list of feature importances
            results = [{'feature': feature, 'attribution': sorted_all_mean_importances[i]} for i, feature in
                       enumerate(sorted_features)]
            return results
        else:
            logging.warning("No importance values were calculated.")
            return None

    # ================ COMPUTING METHODS
    def calculate_actif_mean(self, activation):
        """
           Calculate the mean of absolute activations for each feature.

           Args:
               activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

           Returns:
               mean_activation (np.ndarray): The mean activation for each feature.
           """
        activation_abs = np.abs(activation)  # Take the absolute value of activations
        mean_activation = np.mean(activation_abs, axis=0)  # Compute mean across samples
        return mean_activation  # Return mean activation as the importance

    def calculate_actif_meanstddev(self, activation):
        """
        Calculate the mean and standard deviation of absolute activations for each feature.

        Args:
            activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

        Returns:
            weighted_importance (np.ndarray): Importance as the product of mean and stddev.
            mean_activation (np.ndarray): Mean of the activations.
            std_activation (np.ndarray): Standard deviation of the activations.
        """
        activation_abs = np.abs(activation)  # Take the absolute value of activations
        mean_activation = np.mean(activation_abs, axis=0)  # Compute mean across samples
        std_activation = np.std(activation_abs, axis=0)  # Compute standard deviation across samples
        weighted_importance = mean_activation * std_activation  # Multiply mean by stddev to get importance
        return weighted_importance

    def calculate_actif_inverted_weighted_mean(self, activation):
        """
        Calculate the importance by weighting high mean activations and low variability (stddev).

        Args:
            activation (np.ndarray): The input activations or features, shape (num_samples, num_features).

        Returns:
            adjusted_importance (np.ndarray): Adjusted importance where low variability is rewarded.
        """
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        # Normalize mean and invert normalized stddev
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation))
        inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (
                np.max(std_activation) - np.min(std_activation))

        # Calculate importance as a combination of mean and inverse stddev
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance

    def calculate_actif_robust(self, activations, epsilon=0.01, min_std_threshold=0.01):
        """
        Calculate robust importance where features with high mean activations and low variability are preferred.

        Args:
            activations (np.ndarray): The input activations or features.
            epsilon (float): Small value to prevent division by zero.
            min_std_threshold (float): A threshold to control the impact of stddev.

        Returns:
            adjusted_importance (np.ndarray): Robust importance scores.
        """
        activation_abs = np.abs(activations)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)

        # Normalize the mean and penalize stddev
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                np.max(mean_activation) - np.min(mean_activation) + epsilon)
        transformed_std = np.exp(-std_activation / min_std_threshold)  # Exponentially penalize high stddev
        adjusted_importance = normalized_mean * (1 - transformed_std)

        return adjusted_importance

    '''
    =======================================================================================
    # Established Methods
    =======================================================================================
    '''

    '''
        deepactif
    '''

    def compute_deepactif(self, valid_loader, hook_location, actif_variant):
        """
            deepactif calculation with different layer hooks based on version.
            Args:
                hook_location: Where to hook into the model ('before_lstm', 'after_lstm', 'before_output').
        """
        activations = []
        all_attributions = []
        # self.load_model(self.currentModelName)
        print(f"INFO: Loaded Model: {self.currentModel.__class__.__name__}")

        # Register hooks at different stages based on the version
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]  # For LSTM, handle tuple outputs
                activations.append(output.detach())

            return hook

        # Set hooks based on the chosen version
        if hook_location == 'input':
            # Hook into the input before LSTM (you may need to adjust depending on your model architecture)
            for name, layer in self.currentModel.named_modules():
                if isinstance(layer, torch.nn.LSTM):  # Assuming LSTM is the first main layer
                    layer.register_forward_hook(save_activation(name))  # Use forward hook instead of forward pre-hook
                    break
        elif hook_location == 'lstm':
            # Hook into the output of the LSTM layer
            for name, layer in self.currentModel.named_modules():
                if isinstance(layer, torch.nn.LSTM):
                    layer.register_forward_hook(save_activation(name))
                    break
        elif hook_location == 'penultimate':
            # Hook into fc1 (penultimate layer)
            for name, layer in self.currentModel.named_modules():
                if name == "fc5":  # Specifically target fc1
                    layer.register_forward_hook(save_activation(name))
                    break
        else:
            raise ValueError(f"Unknown hook location: {hook_location}")

        for inputs, _ in valid_loader:
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.currentModel(inputs)

                # Compute output importance for the batch
                if outputs.dim() == 3:
                    output_importance = outputs.mean(dim=1)  # Mean over time steps
                elif outputs.dim() == 2:
                    output_importance = outputs
                else:
                    raise ValueError(f"Unexpected output shape: {outputs.shape}")

                reduced_output_importance = output_importance[:, :len(self.selected_features)]

                # Iterate over samples in the batch
                for i in range(inputs.size(0)):
                    sample_importance = torch.zeros(len(self.selected_features), device=self.device)

                    for activation in activations:
                        if activation.dim() == 3:  # If activations have time steps
                            layer_importance = activation[i].sum(dim=0)[:len(self.selected_features)]
                        elif activation.dim() == 2:
                            layer_importance = activation[i][:len(self.selected_features)]

                        sample_importance += layer_importance * reduced_output_importance[i]

                    # all_attributions.append(sample_importance.cpu().numpy())  # Store sample-level attribution
                    all_attributions.append(sample_importance.detach())  # Kein Wechsel zu NumPy nötig

        all_attributions = np.array(all_attributions)  # Shape: [num_samples, num_features]

        print(f"Final shape of all_attributions: {all_attributions.shape}")

        # Apply ACTIF variant for aggregation
        if actif_variant == 'MEAN':
            importance = self.calculate_actif_mean(all_attributions)
        elif actif_variant == 'MEANSTD':
            importance = self.calculate_actif_meanstddev(all_attributions)
        elif actif_variant == 'INV':
            importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        elif actif_variant == 'PEN':
            importance = self.calculate_actif_robust(all_attributions)
        else:
            raise ValueError(f"Unknown ACTIF variant: {actif_variant}")

        print(f"Calculated importance shape: {importance.shape}")

        # Ensure that the importance variable is a list or array with the same length as the number of features
        if not isinstance(importance, np.ndarray):
            importance = np.array(importance)

        if importance.shape[0] != len(self.selected_features):
            raise ValueError(
                f"ACTIF method returned {importance.shape[0]} importance scores, but {len(self.selected_features)} features are expected."
            )

        # Prepare the results with the aggregated importance
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]

        return results

    '''
       =======================================================================================
       # Utility Functions
       =======================================================================================
       '''

    def model_wrapper(self, model, input_tensor):
        output = model(input_tensor, return_intermediates=False)
        return output.squeeze(-1)

    def calculate_memory_and_execution_time(self, method_func):
        start_time = timeit.default_timer()
        mem_usage, subject_importances = memory_usage((method_func,), retval=True, interval=0.1, timeout=None)
        execution_time = timeit.default_timer() - start_time
        logging.info(f"Execution time: {execution_time} seconds")
        logging.info(f"Memory usage: {max(mem_usage)} MiB")
        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {max(mem_usage)} MiB\n")

        return execution_time, mem_usage, subject_importances

    def save_timing_data(self, method, all_execution_times, all_memory_usages):
        if all_execution_times:
            average_time = sum(all_execution_times) / len(all_execution_times)
            total_time = sum(all_execution_times)
            average_memory = sum(all_memory_usages) / len(all_memory_usages)
            total_memory = sum(all_memory_usages)
            self.timing_data.append({
                'Method': method,
                'Average Execution Time': average_time,
                'Total Execution Time': total_time,
                'Average Memory Usage': average_memory,
                'Total Memory Usage': total_memory
            })
            print(f"Average execution time for {method}: {average_time} seconds")
            print(f"Average memory usage for {method}: {average_memory} MiB \n\n")
            print("=======================================================")

        df_timing = pd.DataFrame(self.timing_data)
        file_path = f"{self.paths['evaluation_metrics_save_path']}/computing_time_and_memory_evaluation_metrics.csv"
        header = not os.path.exists(file_path)
        df_timing.to_csv(file_path, mode='a', index=False, header=header)
        logging.info(f"Appended average execution times to '{file_path}'")

    def sort_importances_based_on_attribution(self, aggregated_importances, method):

        print(f"Before sorting: {len(aggregated_importances)} rows")
        print(aggregated_importances.head())  # Show sample data

        # Convert list of dictionaries to DataFrame if needed
        if isinstance(aggregated_importances, list):
            aggregated_importances = pd.DataFrame(aggregated_importances)

        # Check for valid structure
        if aggregated_importances.empty:
            raise ValueError(f"No feature importances found for method {method}")
        if 'feature' not in aggregated_importances.columns or 'attribution' not in aggregated_importances.columns:
            raise KeyError("The DataFrame does not contain the required 'feature' and 'attribution' columns.")

        print(f"🔍 Before Grouping: {len(aggregated_importances)} rows")  # Should be 34, not 851

        # Ensure there is only one row per feature before sorting
        mean_importances = aggregated_importances.groupby('feature', as_index=False).mean()

        print(f"🔍 After Grouping: {len(mean_importances)} rows")  # Should be 34

        # Sort by attribution values
        sorted_importances = mean_importances.sort_values(by='attribution', ascending=False)

        # Save & return
        self.save_importances_in_file(mean_importances_sorted=sorted_importances, method=method)
        self.feature_importance_results[method] = sorted_importances
        return sorted_importances

        # # Create DataFrame from aggregated importances
        # if isinstance(aggregated_importances, list):
        #     aggregated_importances = pd.DataFrame(aggregated_importances)
        #
        # # Check if DataFrame is empty or doesn't have required columns
        # if aggregated_importances.empty:
        #     raise ValueError(f"No feature importances found for method {method}")
        # if 'feature' not in aggregated_importances.columns or 'attribution' not in aggregated_importances.columns:
        #     raise KeyError("The DataFrame does not contain the required 'feature' and 'attribution' columns.")
        #
        # # Group by 'feature' and compute the mean of the attributions
        # mean_importances = aggregated_importances.groupby('feature')['attribution'].mean().reset_index()
        #
        # # Sort the importances by attribution values
        # sorted_importances = mean_importances.sort_values(by='attribution', ascending=False)
        # self.save_importances_in_file(mean_importances_sorted=sorted_importances, method=method)
        # # Store the sorted importances
        # self.feature_importance_results[method] = sorted_importances
        # return sorted_importances

    def save_importances_in_file(self, mean_importances_sorted, method):
        filename = f"{self.paths['results_folder_path']}/{method}.csv"
        mean_importances_sorted.to_csv(filename, index=False)
        logging.info(f"Saved importances for {method} in {filename}")

    def calculateBaseLine(self, trained_model, valid_loader):
        results = {}

        # Perform cross-validation and get the performance results for each run
        full_feature_performance = self.trainer.cross_validate(num_epochs=500)

        results['Baseline'] = full_feature_performance

        # Save baseline results
        performance_evaluation_file = os.path.join(self.paths["evaluation_metrics_save_path"],
                                                   "performance_evaluation_metrics.txt")

        print("Baseline saved to ", performance_evaluation_file)
        with open(performance_evaluation_file, "a") as file:
            file.write(
                f"Baseline Performance of {self.modelName} on dataset {self.currentDataset.name}: {full_feature_performance}\n")

        print(
            f"Baseline Performance of {self.modelName} on dataset {self.currentDataset.name}: {full_feature_performance}\n")
        return full_feature_performance

    def loadFOVALModel(self, model_path, featureCount=38):
        jsonFile = model_path + '.json'

        with open(jsonFile, 'r') as f:
            hyperparameters = json.load(f)

        model = Foval(feature_count=featureCount, device=self.device)

        return model, hyperparameters

    ###############################

    # def compute_deepactif_ng(self, valid_loader, actif_variant="MEAN"):
    #     # Beispielhafter Aufruf:
    #     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #     # Gehe davon aus, dass trainer.model und selected_features bereits definiert sind.
    #     aggregator = DeepACTIFAggregatorV1(model=self.trainer.model, selected_features=self.selected_features,
    #                                        device=device)
    #     df_importances = aggregator.compute(valid_loader)
    #     print("Final aggregated feature importances:")
    #     print(df_importances)
    #     return df_importances
    #
    # def compute_autodeepactif_full(self, valid_loader):
    #     # Version 2:
    #     # Example Usage:
    #     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #
    #     # Initialize model
    #     model = self.trainer.model
    #
    #     # Run DeepACTIF Analysis on **each sample**
    #     analyzer = AutoDeepACTIF_AnalyzerBackprop(model, self.selected_features, device, use_gaussian_spread=True)
    #
    #     # Um raw Importances mit Zeitinformation zu erhalten, setze return_raw=True:
    #     raw_importances = analyzer.analyze(valid_loader, device, return_raw=True)
    #
    #     # raw_importances: Tensor mit Form (num_samples, timesteps, num_features)
    #     raw_np = raw_importances.cpu().detach().numpy()
    #
    #     # Ermittle den wichtigsten Timestep pro Sample und pro Feature:
    #     important_timesteps = analyzer.most_important_timestep(raw_np)
    #     print("Wichtigster Timestep pro Sample und Feature:")
    #     print(important_timesteps)
    #
    #     # Aggregiere die raw Importances über die Zeit mittels invers gewichteter Methode:
    #     # aggregated_importances = analyzer.aggregate_importance_inv_weighted(raw_np)
    #     # Berechnet den Mittelwert über die Zeitachse (axis=1)
    #     aggregated_importance = np.mean(raw_np, axis=1)
    #     # print("Aggregated importance: ", aggregated_importance.shape)
    #
    #     # Aggregiere über die Zeitdimension (Achse 1)
    #     aggregated_over_time = np.mean(aggregated_importance, axis=1)  # (samples, features)
    #
    #     # Prepare the results with the aggregated importance
    #     results = [{'feature': self.selected_features[i], 'attribution': aggregated_over_time[i]} for i in
    #                range(len(self.selected_features))]
    #     return results
    #
    # def compute_bayesACTIF(self, valid_loader):
    #     # Example Usage:
    #     device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    #
    #     # Angenommen, `full_dataset` ist dein gesamter Dataset.
    #     # Wähle zufällig einen Index-Subset (z.B. 10% der Samples) aus allen Subjects:
    #     # random_samples = len(self.currentDataset.get_random_samples(num_samples=10, subjects=))
    #     # initial_dataloader = DataLoader(random_samples, batch_size=460, shuffle=True)
    #
    #     # Version 1:
    #     # Annahme: trainer.model und selected_features sind bereits definiert.
    #
    #     # Führe die bayessche Analyse durch
    #     bayesian_results = self.bayes_analyzer.compute_bayesACTIF(valid_loader)
    #
    #     # Am Ende kannst du die finalen Priorwerte abrufen:
    #     final_mu, final_sigma = self.bayes_analyzer.mu_prior, self.bayes_analyzer.sigma_prior
    #     print("Final Bayesian mu_prior:", final_mu)
    #     print("Final Bayesian sigma_prior:", final_sigma)
    #
    #     return bayesian_results

    def convert_aggregated_to_dataframe(self, aggregated_importances, selected_features):
        """
        Wandelt ein Aggregatarray der Form (num_samples, num_features) in ein DataFrame um,
        in dem jede Zeile den global gemittelten Attributionswert für ein Feature enthält.
        Dabei wird zuerst über alle Samples gemittelt, sodass ein Vektor der Länge num_features entsteht,
        und dieser Vektor wird dann in ein DataFrame mit den Spalten "feature" und "attribution" überführt.

        Args:
            aggregated_importances (np.ndarray): Array der Form (num_samples, num_features)
            selected_features (list): Liste der Feature-Namen (Länge num_features)

        Returns:
            pd.DataFrame: DataFrame mit Spalten "feature" und "attribution"
        """
        # Globaler Mittelwert über alle Samples pro Feature:
        global_importance = aggregated_importances.mean(axis=0)  # Shape: (num_features,)
        df_final = pd.DataFrame({
            "feature": selected_features,
            "attribution": global_importance
        })
        return df_final

    def compute_deepactif_(self, valid_loader, method):

        if method == '1_DeepACTIF_FULL_OLD':
            raw_np = self.analyzer.analyze(dataloader=valid_loader)

            # Ensure conversion to NumPy
            raw_np = raw_np.cpu().numpy()  # Shape: (num_samples, timesteps, num_features)

            # Aggregate over timesteps
            aggregated_over_time = np.mean(raw_np, axis=1)  # Shape: (num_samples, num_features)

            # Aggregate over samples to get one value per feature
            aggregated_features = np.mean(aggregated_over_time, axis=0)  # Shape: (num_features,)

            # Prepare the final feature importance results
            results = [{'feature': self.selected_features[i], 'attribution': aggregated_features[i]}
                       for i in range(len(self.selected_features))]

        elif method == '2_DeepACTIF_Forward_OLD':
            results = self.analyzer.analyze(dataloader=valid_loader, device=self.device)
        elif method == '3_AutoML_DeepACTIF':

            # BACKUP WORKING
            # # Um raw Importances mit Zeitinformation zu erhalten, setze return_raw=True:
            # raw_importances = self.analyzer.analyze(valid_loader, self.device, return_raw=True)
            #
            # # raw_importances: Tensor mit Form (num_samples, timesteps, num_features)
            # raw_np = raw_importances.cpu().detach().numpy()
            #
            # # SEPARATE: Ermittle den wichtigsten Timestep pro Sample und pro Feature:
            # important_timesteps = self.analyzer.most_important_timestep(raw_np)
            # print("Wichtigster Timestep pro Sample und Feature:")
            # print(important_timesteps)
            #
            # # Aggregiere die raw Importances über die Zeit mittels invers gewichteter Methode:
            # # aggregated_importances = analyzer.aggregate_importance_inv_weighted(raw_np)
            # # ODER
            # # Berechnet den Mittelwert über die Zeitachse (axis=1)
            # aggregated_importance = np.mean(raw_np, axis=1)
            #
            # # Aggregiere über die Zeitdimension (Achse 1)
            # aggregated_over_time = np.mean(aggregated_importance, axis=1)  # (samples, features)
            #
            # # Prepare the results with the aggregated importance
            # results = [{'feature': self.selected_features[i], 'attribution': aggregated_over_time[i]} for i in
            #            range(len(self.selected_features))]

            # Tidy-Up Version
            # Um raw Importances mit Zeitinformation zu erhalten, setze return_raw=True:
            raw_importances = self.analyzer.analyze(valid_loader, self.device, return_raw=True)

            # SEPARATE: Ermittle den wichtigsten Timestep pro Sample und pro Feature:
            important_timesteps = self.analyzer.most_important_timestep(raw_importances.cpu().detach().numpy())
            print("Wichtigster Timestep pro Sample und Feature:")
            print(important_timesteps)

            # Aggregiere die raw Importances über die Zeit mittels invers gewichteter Methode:
            # INV or NONE: inverted weighted mean or simply mean across time
            aggregated_importances = self.analyzer.aggregate_importances(raw_importances.cpu().detach().numpy(), method='INV')

            # Aggregiere über die Zeitdimension (Achse 1)
            aggregated_over_time = np.mean(aggregated_importances, axis=1)  # (samples, features)

            # Prepare the results with the aggregated importance
            results = [{'feature': self.selected_features[i], 'attribution': aggregated_over_time[i]} for i in
                       range(len(self.selected_features))]

        elif method == '4_DeepACTIF_AggregatorV1':
            results = self.analyzer.compute(valid_loader=valid_loader)
        elif method == '5_BayesianInference':
            results = self.analyzer.compute_bayesACTIF(valid_loader=valid_loader)
        elif method == '6_BayesianInference_Optim':
            # Führe die bayessche Analyse durch
            results = self.analyzer.compute_bayesACTIF(valid_loader)

            # Am Ende kannst du die finalen Priorwerte abrufen:
            final_mu, final_sigma = self.analyzer.mu_prior, self.analyzer.sigma_prior
            print("Final Bayesian mu_prior:", final_mu)
            print("Final Bayesian sigma_prior:", final_sigma)

        else:
            print("Method unknwon")

        return results


'''
Varianten:
1. Single Layer, No backprop, single-Interpolation back ///// ACTIF
2. Full Model, No backprop, single-Interpolation back   ///// DeepACTIF    
3. Full Model, Backpropagation by interpolating all layers
4. 
5. Full Model. Bayesian Updating, no backprop
6. FUll Model, Bayesian Updating, InitialDataset priors, weight attributions with uncertainty

1. Feature Importance Calculation Method
	•   USE BACKPROPAGATION: A_AutoDeepACTIF_AnalyzerBackprop, AutoDeepACTIF_AnalyzerBackprop
	•	USE ONLY FORWARD PASS:  DeepACTIFAnalyzerForward_2, DeepACTIFAggregatorV1
	•	USE PROBABILISTIC UPDATES: Bayesian versions (BayesianFeatureImportance_orig, BayesianFeatureImportance) use probabilistic updates instead of raw activations.

2. Layer Hooking Strategy
	•	Backpropagation versions: Use forward hooks to capture activations from Linear, LSTM, and Pooling layers for backpropagation 
	    (e.g., AutoDeepACTIF_AnalyzerBackprop).
	•	Forward-based versions: Use sequential forward passes through the network and directly compute importance (e.g., DeepACTIFAggregatorV1).
	•	Bayesian versions: Extend this further with Bayesian updates per feature, rather than just summing importances.

3. Temporal Feature Handling
	•	Backpropagation approaches: Spread importance backwards through pooling and LSTM layers (e.g., backpropagate_max_pooling applies Gaussian spreading).
	•	Forward-based approaches: Compute per-layer importance and interpolate along features (e.g., DeepACTIFAnalyzerForward_2).
	•	Bayesian methods: Aggregate feature importance over time using median absolute deviation (MAD) and probabilistic updates.

4. Bayesian Updates & Uncertainty Estimation
	•	BayesianFeatureImportance_orig: Uses a simple Bayesian update method but doesn’t integrate prior learning well.
	•	BayesianFeatureImportance: Implements a more robust Bayesian updating rule, initializing with a small dataset to set priors (mu_prior, sigma_prior).
	•	Introduces adaptive learning rate decay (initial_alpha, decay_rate).

5. Output Format & Interpretation
	•	Some versions return raw temporal importance per timestep (return_raw=True), allowing further analysis of time dependencies.
	•	Aggregation methods (DeepACTIFAggregatorV1, AutoDeepACTIF_AnalyzerBackprop) produce SHAP-like outputs mapping importance to feature names.
	•	Bayesian ACTIF (compute_bayesACTIF) provides uncertainty-aware feature attributions, allowing ranking based on confidence.

'''
