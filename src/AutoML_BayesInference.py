import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# --- Singleton-Metaklasse ---
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# --- BayesianFeatureImportance Klasse als Singleton ---
class BayesianFeatureImportance(metaclass=Singleton):
    def __init__(self, model, features, device, use_gaussian_spread=True):
        """
        Initialisiert den BayesianFeatureImportance-Analyzer.

        Args:
            model (torch.nn.Module): Das trainierte Modell.
            features (list): Liste der Feature-Namen.
            device (str): Das Gerät, auf dem die Analyse durchgeführt wird.
            use_gaussian_spread (bool): Wählt zwischen Gaußschem Spread und direkter Zuordnung.
        """
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.use_gaussian_spread = use_gaussian_spread

        self.activations = {}
        self.layer_types = {}
        self.max_indices = {}

        # Bayesian-Prior initialisieren: Wir starten mit mu=0 und sigma=1 für jedes Feature.
        self.mu_prior = np.zeros(len(self.selected_features))
        self.sigma_prior = np.ones(len(self.selected_features))
        # Zusätzlich führen wir eine laufende Beobachtungsvarianz (obs_var) ein,
        # die über die Batches hinweg als moving average aktualisiert wird.
        self.obs_var = np.ones(len(self.selected_features))
        self.num_batches = 0

    def register_hooks(self):
        def hook_fn(name, module):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach()
                self.layer_types[name] = type(module).__name__
                if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                    _, max_idx = torch.max(output, dim=1)
                    self.max_indices[name] = max_idx

            return hook

        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
                layer.register_forward_hook(hook_fn(name, layer))

    def backpropagate_max_pooling(self, importance, activations, layer_name):
        print(f"🔄 Backpropagating through Max Pooling layer: {layer_name}")
        if layer_name not in self.max_indices:
            raise ValueError(f"🚨 No max_indices found for {layer_name}.")
        max_indices = self.max_indices[layer_name]
        batch_size, timesteps, num_features = activations.shape
        expanded_importance = torch.zeros_like(activations)
        if self.use_gaussian_spread:
            for b in range(batch_size):
                for f in range(num_features):
                    max_t = max_indices[b, f].item()
                    for t in range(timesteps):
                        expanded_importance[b, t, f] = (
                                importance[b, f] *
                                torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
                                                                device=importance.device) - max_t) / 2.0) ** 2)
                        )
        else:
            for b in range(batch_size):
                for f in range(num_features):
                    expanded_importance[b, max_indices[b, f], f] = importance[b, f]
        print(f"✅ Final importance shape after Max Pooling: {expanded_importance.shape}")
        return expanded_importance

    def forward_pass_with_importance(self, inputs, return_raw=False):
        self.model.eval()
        activations = inputs
        T = inputs.shape[1]
        accumulated_importance = None

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.LSTM):
                activations, _ = layer(activations)
                layer_importance = torch.abs(activations)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)
                if activations.dim() == 2:
                    layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
                else:
                    layer_importance = torch.abs(activations)
            else:
                continue

            N, T_current, orig_features = layer_importance.shape
            reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            interpolated = F.interpolate(reshaped, size=len(self.selected_features),
                                         mode="linear", align_corners=False)
            interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
            if accumulated_importance is None:
                accumulated_importance = interpolated_importance
            else:
                if accumulated_importance.shape != interpolated_importance.shape:
                    raise ValueError(
                        f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}")
                accumulated_importance += interpolated_importance

        if return_raw:
            return accumulated_importance
        else:
            aggregated_importance = accumulated_importance.mean(dim=1)
            return aggregated_importance

    def analyze(self, dataloader, return_raw=False):
        all_importances = []
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)
            importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
            if importance is None or torch.isnan(importance).any():
                print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
                continue
            all_importances.append(importance)
            cumulative = torch.cat(all_importances, dim=0)
            print(f"Nach Batch {batch_idx}: Kumulierte Importances-Shape: {cumulative.shape}")
        if not all_importances:
            raise ValueError("❌ No valid feature importances calculated!")
        concatenated = torch.cat(all_importances, dim=0)
        return concatenated if return_raw else concatenated

    def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
        activation_abs = np.abs(activation)
        mean_activation = np.mean(activation_abs, axis=0)
        std_activation = np.std(activation_abs, axis=0)
        normalized_mean = (mean_activation - np.min(mean_activation)) / (
                    np.max(mean_activation) - np.min(mean_activation) + epsilon)
        inverse_normalized_std = 1 - ((std_activation - np.min(std_activation)) / (
                    np.max(std_activation) - np.min(std_activation) + epsilon))
        adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
        return adjusted_importance

    def most_important_timestep(self, all_attributions):
        return np.argmax(np.abs(all_attributions), axis=1)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
        mean_activation = np.mean(np.abs(all_attributions), axis=1)
        std_activation = np.std(np.abs(all_attributions), axis=1)
        min_mean = np.min(mean_activation, axis=1, keepdims=True)
        max_mean = np.max(mean_activation, axis=1, keepdims=True)
        normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
        min_std = np.min(std_activation, axis=1, keepdims=True)
        max_std = np.max(std_activation, axis=1, keepdims=True)
        normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
        inverse_normalized_std = 1 - normalized_std
        aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
        return aggregated_importance

    def aggregate_importances_over_samples(self, sample_importances):
        if isinstance(sample_importances, torch.Tensor):
            sample_importances = sample_importances.cpu().detach().numpy()
        print("Input to aggregate_importances_over_samples, shape:", sample_importances.shape)
        global_importance = np.mean(sample_importances, axis=0)
        print("Global importance after aggregation:", global_importance.shape, global_importance)
        if global_importance.shape[0] != len(self.selected_features):
            raise ValueError(f"Expected {len(self.selected_features)} features, but got {global_importance.shape[0]}.")
        df_final = pd.DataFrame({
            "feature": self.selected_features,
            "attribution": global_importance
        })
        print("Final DataFrame shape:", df_final.shape)
        return df_final

    def aggregate_and_return(self, sample_importances):
        if (isinstance(sample_importances, list) and len(sample_importances) == 0) or \
                (isinstance(sample_importances, torch.Tensor) and sample_importances.numel() == 0):
            raise ValueError("❌ No feature importances calculated from the data.")
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
                           for imp in sample_importances]
        all_attributions = np.vstack(all_importances)
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in all_attributions!")
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)
        print("All Attributions: ", all_attributions)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]}
                   for i in range(len(self.selected_features))]
        print("Results from Class: ", results)
        return results

    def backpropagate_linear(self, importance, activations, layer_name):
        print(f"🔄 Backpropagating through Linear layer: {layer_name}")
        if importance.dim() == 1:
            importance = importance.unsqueeze(0)
        if importance.dim() == 2:
            importance = importance.unsqueeze(1)
        target_size = (activations.shape[-1],)
        print(f"⚠ Interpolating Linear layer from {importance.shape[-1]} to {target_size}")
        importance = F.interpolate(importance, size=target_size, mode="linear", align_corners=False).squeeze(1)
        print(f"✅ Backpropagated Linear Importance Shape: {importance.shape}")
        return importance

    def backpropagate_lstm(self, importance, activations, layer_name):
        print(f"🔄 Backpropagating through LSTM layer: {layer_name}")
        batch_size, timesteps, num_features = activations.shape
        if importance.dim() == 2:
            importance = importance.unsqueeze(1)
        elif importance.dim() == 3 and importance.shape[1] == num_features:
            importance = importance.permute(0, 2, 1)
        print(f"✅ Adjusted importance shape before interpolation: {importance.shape}")
        importance = importance.unsqueeze(1)
        target_size = (timesteps, len(self.selected_features))
        print(f"✅ Interpolating to match input shape {target_size}")
        importance = F.interpolate(importance, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
        print(f"✅ Final importance shape after LSTM backpropagation: {importance.shape}")
        return importance

    def weighted_interpolation(self, importance, activations):
        batch_size, timesteps, num_features = activations.shape
        max_indices = activations.abs().max(dim=1).indices
        spread_importance = torch.zeros_like(activations)
        for i in range(batch_size):
            max_t = max_indices[i].item()
            time_range = torch.arange(timesteps, device=importance.device)
            gaussian_weights = torch.exp(-((time_range - max_t) ** 2) / (2 * (timesteps / 5) ** 2))
            gaussian_weights = gaussian_weights / gaussian_weights.sum()
            spread_importance[i] = gaussian_weights[:, None] * importance[i]
        print(f"✅ Final importance shape after max-pooling backpropagation: {spread_importance.shape}")
        return spread_importance

    def update_bayesian(self, sample_importance, sigma_obs):
        """
        Aktualisiert die Priorverteilungen (mu_prior, sigma_prior) für jedes Feature anhand einer neuen Beobachtung.

        Args:
            sample_importance (np.ndarray): Vektor der Form (num_features,) mit den gemessenen Importances.
            sigma_obs: Entweder ein Skalar oder ein Vektor (np.ndarray) mit den beobachteten Standardabweichungen pro Feature.
        """
        for i in range(len(self.selected_features)):
            prior_var = self.sigma_prior[i] ** 2
            if np.isscalar(sigma_obs):
                obs_var = sigma_obs ** 2
            else:
                obs_var = sigma_obs[i] ** 2
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
            posterior_mean = posterior_var * (self.mu_prior[i] / prior_var + sample_importance[i] / obs_var)
            self.mu_prior[i] = posterior_mean
            self.sigma_prior[i] = np.sqrt(posterior_var)
        print("Updated Bayesian priors:")
        print("Mu:", self.mu_prior)
        print("Sigma:", self.sigma_prior)

    def analyze_bayesian(self, dataloader, device):
        """
        Führt einen vollständigen Durchlauf über den Datensatz durch und aktualisiert
        die bayesschen Prior-Wahrscheinlichkeitsverteilungen (mu_prior, sigma_prior) für jedes Feature
        anhand der pro Sample aggregierten Importances.

        Args:
            dataloader: DataLoader des Datensatzes.
            device (str): z.B. "cuda", "mps" oder "cpu".

        Returns:
            Tuple[np.ndarray, np.ndarray]: Die finalen mu_prior und sigma_prior.
        """
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, dtype=torch.float32)
            sample_importances = self.forward_pass_with_importance(batch, return_raw=False)
            sample_importances_np = sample_importances.cpu().detach().numpy()  # (batch, num_features)
            # Berechne für diesen Batch die feature-spezifische Beobachtungsvarianz
            batch_obs_var = np.std(sample_importances_np, axis=0)
            # Optional: Aktualisiere eine laufende Beobachtungsvarianz
            self.num_batches += 1
            self.obs_var = (self.obs_var * (self.num_batches - 1) + batch_obs_var) / self.num_batches
            print(f"Batch {batch_idx}: Beobachtungsvarianz (sigma_obs) = {self.obs_var}")
            for sample_imp in sample_importances_np:
                self.update_bayesian(sample_imp, sigma_obs=self.obs_var)
        print("Final Bayesian mu_prior:", self.mu_prior)
        print("Final Bayesian sigma_prior:", self.sigma_prior)
        return self.mu_prior, self.sigma_prior

    # def compute_bayesACTIF(self, valid_loader):
    #     """
    #     Führt die bayessche DeepACTIF-Analyse durch: Für jedes Sample wird die aggregierte Importance berechnet
    #     und dann bayessch aktualisiert.
    #
    #     Returns:
    #         dict: Dictionary mit 'mu' (Posterior-Mittelwerte) und 'sigma' (Posterior-Unsicherheiten) pro Feature.
    #     """
    #     mu_post, sigma_post = self.analyze_bayesian(valid_loader, self.device)
    #     results = [{'feature': self.selected_features[i],
    #                 'attribution': mu_post[i],
    #                 'uncertainty': sigma_post[i]} for i in range(len(self.selected_features))]
    #     print("Bayesian ACTIF Results:")
    #     for r in results:
    #         print(r)
    #     return results

    def compute_bayesACTIF(self, valid_loader):
        mu_post, sigma_post = self.analyze_bayesian(valid_loader, self.device)

        # Erstelle eine Liste von Dictionaries, aber nur mit den relevanten Keys
        results = [{'feature': self.selected_features[i], 'attribution': mu_post[i]} for i in
                   range(len(self.selected_features))]

        # Debugging: Prüfen, ob "feature" und "attribution" enthalten sind
        print("Bayesian ACTIF Results (first 5 entries):", results[:5])

        return results

    def aggregate_and_return(self, sample_importances):
        if (isinstance(sample_importances, list) and len(sample_importances) == 0) or \
                (isinstance(sample_importances, torch.Tensor) and sample_importances.numel() == 0):
            raise ValueError("❌ No feature importances calculated from the data.")
        all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
                           for imp in sample_importances]
        all_attributions = np.vstack(all_importances)
        if np.isnan(all_attributions).any():
            print("⚠️ Warning: NaN values detected in all_attributions!")
            all_attributions = np.nan_to_num(all_attributions, nan=0.0)
        print("All Attributions: ", all_attributions)
        importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]}
                   for i in range(len(self.selected_features))]
        print("Results from Class: ", results)
        return results

    def model_wrapper(self, model, input_tensor):
        output = model(input_tensor, return_intermediates=False)
        return output.squeeze(-1)

    def calculate_memory_and_execution_time(self, method_func):
        import timeit
        from memory_profiler import memory_usage
        start_time = timeit.default_timer()
        mem_usage, subject_importances = memory_usage((method_func,), retval=True, interval=0.1, timeout=None)
        execution_time = timeit.default_timer() - start_time
        print(f"Execution time: {execution_time} seconds")
        print(f"Memory usage: {max(mem_usage)} MiB\n")
        return execution_time, mem_usage, subject_importances

    def save_timing_data(self, method, all_execution_times, all_memory_usages):
        import os
        df_timing = pd.DataFrame({
            'Method': [method],
            'Average Execution Time': [sum(all_execution_times) / len(all_execution_times)],
            'Total Execution Time': [sum(all_execution_times)],
            'Average Memory Usage': [sum(all_memory_usages) / len(all_memory_usages)],
            'Total Memory Usage': [sum(all_memory_usages)]
        })
        file_path = os.path.join(self.paths['evaluation_metrics_save_path'],
                                 "computing_time_and_memory_evaluation_metrics.csv")
        header = not os.path.exists(file_path)
        df_timing.to_csv(file_path, mode='a', index=False, header=header)
        print(f"Saved timing data to '{file_path}'")

    def sort_importances_based_on_attribution(self, aggregated_importances, method):
        # Erzeuge DataFrame, falls aggregated_importances als Liste vorliegt
        if isinstance(aggregated_importances, list):
            aggregated_importances = pd.DataFrame(aggregated_importances)
        # Überprüfe, ob die erforderlichen Spalten vorhanden sind
        if aggregated_importances.empty:
            raise ValueError(f"No feature importances found for method {method}")
        if 'feature' not in aggregated_importances.columns or 'attribution' not in aggregated_importances.columns:
            raise KeyError("The DataFrame does not contain the required 'feature' and 'attribution' columns.")
        # Gruppiere nach 'feature' und bilde den Mittelwert der Attributionswerte
        mean_importances = aggregated_importances.groupby('feature')['attribution'].mean().reset_index()
        sorted_importances = mean_importances.sort_values(by='attribution', ascending=False)
        self.save_importances_in_file(sorted_importances, method)
        self.feature_importance_results[method] = sorted_importances
        return sorted_importances

    def save_importances_in_file(self, mean_importances_sorted, method):
        import os
        filename = os.path.join(self.paths['results_folder_path'], f"{method}.csv")
        mean_importances_sorted.to_csv(filename, index=False)
        print(f"Saved importances for {method} in {filename}")

    def calculateBaseLine(self, trained_model, valid_loader):
        full_feature_performance = self.trainer.cross_validate(num_epochs=500)
        import os
        performance_evaluation_file = os.path.join(self.paths["evaluation_metrics_save_path"],
                                                   "performance_evaluation_metrics.txt")
        print("Baseline saved to ", performance_evaluation_file)
        with open(performance_evaluation_file, "a") as file:
            file.write(
                f"Baseline Performance of {self.model.__class__.__name__} on dataset {self.currentDataset.name}: {full_feature_performance}\n")
        print(f"Baseline Performance: {full_feature_performance}\n")
        return full_feature_performance

    def loadFOVALModel(self, model_path, featureCount=38):
        import json, os
        jsonFile = model_path + '.json'
        with open(jsonFile, 'r') as f:
            hyperparameters = json.load(f)
        from src.models.FOVAL.foval import Foval
        model = Foval(feature_count=featureCount, device=self.device)
        return model, hyperparameters