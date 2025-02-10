import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


#
# Zuerst definieren wir einen Singleton-Metaklassen-Mechanismus:
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Erzeuge eine neue Instanz, wenn sie noch nicht existiert
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


#
#
# # Beispiel: BayesianFeatureImportance als Singleton
# class BayesianFeatureImportance(metaclass=Singleton):
#     def __init__(self, model, features, device, use_gaussian_spread=True):
#         self.model = model
#         self.selected_features = features if features is not None else []
#         self.device = device
#         self.use_gaussian_spread = use_gaussian_spread
#         self.activations = {}
#         self.layer_types = {}
#         self.max_indices = {}
#         self.register_hooks()
#
#     def register_hooks(self):
#         def hook_fn(name, module):
#             def hook(module, input, output):
#                 if isinstance(output, tuple):
#                     output = output[0]
#                 self.activations[name] = output.detach()
#                 self.layer_types[name] = type(module).__name__
#                 if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                     _, max_idx = torch.max(output, dim=1)
#                     self.max_indices[name] = max_idx
#
#             return hook
#
#         for name, layer in self.model.named_modules():
#             if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                 layer.register_forward_hook(hook_fn(name, layer))
#
#     def forward_pass_with_importance(self, inputs, return_raw=False):
#         self.model.eval()
#         activations = inputs
#         T = inputs.shape[1]
#         accumulated_importance = None
#
#         for name, layer in self.model.named_children():
#             if isinstance(layer, nn.LSTM):
#                 activations, _ = layer(activations)
#                 layer_importance = torch.abs(activations)
#             elif isinstance(layer, nn.Linear):
#                 activations = layer(activations)
#                 if activations.dim() == 2:
#                     layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
#                 else:
#                     layer_importance = torch.abs(activations)
#             else:
#                 continue
#
#             # Interpolation der Feature-Dimension: von orig_features zu len(selected_features)
#             N, T_current, orig_features = layer_importance.shape
#             reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
#             interpolated = F.interpolate(reshaped, size=len(self.selected_features), mode="linear", align_corners=False)
#             interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
#
#             if accumulated_importance is None:
#                 accumulated_importance = interpolated_importance
#             else:
#                 if accumulated_importance.shape != interpolated_importance.shape:
#                     raise ValueError(
#                         f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}"
#                     )
#                 accumulated_importance += interpolated_importance
#
#         if return_raw:
#             return accumulated_importance
#         else:
#             aggregated_importance = accumulated_importance.mean(dim=1)
#             return aggregated_importance
#
#     def analyze(self, dataloader, device, return_raw=False, live_visualizer=None, update_interval=5):
#         """
#         Berechnet die Feature-Importances über den gesamten Datensatz.
#         Optional werden, falls live_visualizer übergeben wird, in regelmäßigen Abständen die aktuellen
#         Werte an den Visualizer übergeben.
#         Args:
#             dataloader: DataLoader mit den Samples.
#             device: z.B. "cuda" oder "cpu".
#             return_raw (bool): Falls True, wird der Tensor mit Zeitinformation (shape: [N, T, features]) zurückgegeben.
#             live_visualizer: Optionaler LiveVisualizer (z. B. Instanz der oben gezeigten Klasse), der Updates erhält.
#             update_interval (int): Aktualisierung erfolgt alle update_interval Batches.
#         Returns:
#             Tensor oder Array der aggregierten Importances (je nach return_raw).
#         """
#         all_importances = []
#         batch_count = 0
#         for batch_idx, batch in enumerate(dataloader):
#             if isinstance(batch, (list, tuple)):
#                 batch = batch[0]
#             batch = batch.to(device, dtype=torch.float32)
#             importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
#             if importance is None or torch.isnan(importance).any():
#                 print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
#                 continue
#             all_importances.append(importance)
#             batch_count += 1
#
#             if live_visualizer is not None and (batch_count % update_interval == 0):
#                 # Berechne aggregierte Importances und Unsicherheiten für den aktuellen Batch
#                 aggregated = importance.mean(dim=1)  # shape: (batch_size, num_features)
#                 uncert = importance.std(dim=1)  # shape: (batch_size, num_features)
#                 # Übergib die aktuellen Werte an den Visualizer (wir zeigen hier z. B. den ersten Sample des aktuellen Batches)
#                 live_visualizer.update(
#                     raw_importances=importance.cpu().detach().numpy(),
#                     aggregated_importance=aggregated.cpu().detach().numpy(),
#                     uncertainty=uncert.cpu().detach().numpy(),
#                     sample_idx=0,
#                     iteration=batch_count
#                 )
#
#         if not all_importances:
#             raise ValueError("❌ No valid feature importances calculated!")
#         all_importances = torch.cat(all_importances, dim=0)
#         return all_importances
#
#     def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
#         mean_activation = np.mean(np.abs(activation), axis=1)
#         std_activation = np.std(np.abs(activation), axis=1)
#         min_mean = np.min(mean_activation, axis=1, keepdims=True)
#         max_mean = np.max(mean_activation, axis=1, keepdims=True)
#         normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
#         min_std = np.min(std_activation, axis=1, keepdims=True)
#         max_std = np.max(std_activation, axis=1, keepdims=True)
#         normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
#         inverse_normalized_std = 1 - normalized_std
#         aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
#         return aggregated_importance
#
#     def most_important_timestep(self, all_attributions):
#         return np.argmax(np.abs(all_attributions), axis=1)
#
#     def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
#         mean_activation = np.mean(np.abs(all_attributions), axis=1)
#         std_activation = np.std(np.abs(all_attributions), axis=1)
#         min_mean = np.min(mean_activation, axis=1, keepdims=True)
#         max_mean = np.max(mean_activation, axis=1, keepdims=True)
#         normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
#         min_std = np.min(std_activation, axis=1, keepdims=True)
#         max_std = np.max(std_activation, axis=1, keepdims=True)
#         normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
#         inverse_normalized_std = 1 - normalized_std
#         aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
#         return aggregated_importance

#
# # Beispiel: BayesianFeatureImportance als Singleton
# class BayesianFeatureImportance(metaclass=Singleton):
#     def __init__(self, model, features, device, use_gaussian_spread=True):
#         """
#         Initialize the analyzer.
#
#         Args:
#             model (torch.nn.Module): The trained model.
#             features (list): List of feature names.
#             device (str): The device to run the analysis on.
#             use_gaussian_spread (bool): Flag to toggle between Gaussian spreading and equal distribution.
#         """
#         self.use_gaussian_spread = use_gaussian_spread
#         self.use_gaussian_spread = True
#         self.max_indices = None
#         self.layer_types = None
#         self.activations = None
#         self.model = model
#         self.selected_features = features if features is not None else []
#         self.device = device
#         self.alpha = 0.1
#         self.live_visualizer = None
#         # Beispielhafte Initialisierung der Priorwerte (diese werden dann über alle Subjects hinweg aktualisiert)
#         self.mu_prior = np.zeros(len(self.selected_features))
#         self.sigma_prior = np.ones(len(self.selected_features))
#
#         # Bayesian-Prior für jedes Feature (als Vektor, Länge = Anzahl Features)
#         self.mu_prior = np.zeros(len(self.selected_features))
#         self.sigma_prior = np.ones(len(self.selected_features))
#
#     def register_hooks(self):
#         def hook_fn(name, module):
#             def hook(module, input, output):
#                 if isinstance(output, tuple):
#                     output = output[0]
#                 self.activations[name] = output.detach()
#                 self.layer_types[name] = type(module).__name__
#                 if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                     _, max_idx = torch.max(output, dim=1)
#                     self.max_indices[name] = max_idx
#
#             return hook
#
#         for name, layer in self.model.named_modules():
#             if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                 layer.register_forward_hook(hook_fn(name, layer))
#
#     def backpropagate_max_pooling(self, importance, activations, layer_name):
#         print(f"🔄 Backpropagating through Max Pooling layer: {layer_name}")
#         if layer_name not in self.max_indices:
#             raise ValueError(f"🚨 No max_indices found for {layer_name}.")
#         max_indices = self.max_indices[layer_name]
#         batch_size, timesteps, num_features = activations.shape
#         expanded_importance = torch.zeros_like(activations)
#         if self.use_gaussian_spread:
#             for b in range(batch_size):
#                 for f in range(num_features):
#                     max_t = max_indices[b, f].item()
#                     for t in range(timesteps):
#                         expanded_importance[b, t, f] = (
#                                 importance[b, f] * torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
#                                                                                    device=importance.device) - max_t) / 2.0) ** 2)
#                         )
#         else:
#             for b in range(batch_size):
#                 for f in range(num_features):
#                     expanded_importance[b, max_indices[b, f], f] = importance[b, f]
#         print(f"✅ Final importance shape after Max Pooling: {expanded_importance.shape}")
#         return expanded_importance
#
#     def forward_pass_with_importance(self, inputs, return_raw=False):
#         self.model.eval()
#         activations = inputs
#         T = inputs.shape[1]
#         accumulated_importance = None
#
#         for name, layer in self.model.named_children():
#             if isinstance(layer, nn.LSTM):
#                 activations, _ = layer(activations)
#                 layer_importance = torch.abs(activations)  # (batch, L, hidden)
#             elif isinstance(layer, nn.Linear):
#                 activations = layer(activations)
#                 if activations.dim() == 2:
#                     layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
#                 else:
#                     layer_importance = torch.abs(activations)
#             else:
#                 continue
#
#             # Interpolation entlang der Feature-Dimension
#             N, T_current, orig_features = layer_importance.shape
#             reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
#             interpolated = F.interpolate(reshaped, size=len(self.selected_features),
#                                          mode="linear", align_corners=False)
#             interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
#             if accumulated_importance is None:
#                 accumulated_importance = interpolated_importance
#             else:
#                 if accumulated_importance.shape != interpolated_importance.shape:
#                     raise ValueError(
#                         f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}")
#                 accumulated_importance += interpolated_importance
#
#         if return_raw:
#             return accumulated_importance
#         else:
#             aggregated_importance = accumulated_importance.mean(dim=1)
#             return aggregated_importance
#
#     # def forward_pass_with_importance(self, inputs, return_raw=False):
#     #     self.model.eval()
#     #     T = inputs.shape[1]
#     #     accumulated_importance = None
#     #     for name, layer in self.model.named_children():
#     #         if isinstance(layer, nn.LSTM):
#     #             activations, _ = layer(inputs)
#     #             layer_importance = torch.abs(activations)
#     #         elif isinstance(layer, nn.Linear):
#     #             inputs = layer(inputs)
#     #             if inputs.dim() == 2:
#     #                 layer_importance = torch.abs(inputs).unsqueeze(1).expand(-1, T, -1)
#     #             else:
#     #                 layer_importance = torch.abs(inputs)
#     #         else:
#     #             continue
#     #         N, T_current, orig_features = layer_importance.shape
#     #         reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
#     #         interpolated = F.interpolate(reshaped, size=len(self.selected_features), mode="linear", align_corners=False)
#     #         interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
#     #         if accumulated_importance is None:
#     #             accumulated_importance = interpolated_importance
#     #         else:
#     #             if accumulated_importance.shape != interpolated_importance.shape:
#     #                 raise ValueError(
#     #                     f"Shape mismatch: {accumulated_importance.shape} vs. {interpolated_importance.shape}")
#     #             accumulated_importance += interpolated_importance
#     #     if return_raw:
#     #         return accumulated_importance
#     #     else:
#     #         aggregated_importance = accumulated_importance.mean(dim=1)
#     #         return aggregated_importance
#
#     # def analyze(self, dataloader, device, return_raw=False, live_visualizer=None):
#     #     all_importances = []
#     #     for batch_idx, batch in enumerate(dataloader):
#     #         if isinstance(batch, (list, tuple)):
#     #             batch = batch[0]
#     #         batch = batch.to(device, dtype=torch.float32)
#     #         importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
#     #         if importance is None or torch.isnan(importance).any():
#     #             print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
#     #             continue
#     #         all_importances.append(importance)
#     #         # Hier: Live-Update, wenn ein LiveVisualizer übergeben wurde
#     #         if live_visualizer is not None:
#     #             concatenated = torch.cat(all_importances,
#     #                                      dim=0)  # (num_samples, timesteps, num_features) falls return_raw=True
#     #             if return_raw:
#     #                 agg = concatenated.mean(dim=1)  # (num_samples, num_features)
#     #                 uncert = concatenated.std(dim=1)  # (num_samples, num_features)
#     #             else:
#     #                 agg = concatenated  # Bereits aggregiert
#     #                 uncert = torch.zeros_like(agg)
#     #             live_visualizer.update(
#     #                 raw_importances=concatenated.cpu().numpy(),
#     #                 aggregated_importance=agg.cpu().numpy(),
#     #                 uncertainty=uncert.cpu().numpy(),
#     #                 sample_idx=0
#     #             )
#     #     if not all_importances:
#     #         raise ValueError("❌ No valid feature importances calculated!")
#     #     return torch.cat(all_importances, dim=0)
#
#     def analyze(self, dataloader, return_raw=False):
#         all_importances = []
#         for batch_idx, batch in enumerate(dataloader):
#             if isinstance(batch, (list, tuple)):
#                 batch = batch[0]
#             batch = batch.to(self.device, dtype=torch.float32)
#             importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
#             if importance is None or torch.isnan(importance).any():
#                 print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
#                 continue
#             all_importances.append(importance)
#             # Hier: Live-Update, wenn ein LiveVisualizer übergeben wurde
#             if self.live_visualizer is not None:
#                 concatenated = torch.cat(all_importances, dim=0)  # (num_samples, timesteps, num_features) falls return_raw=True
#                 if return_raw:
#                     agg = concatenated.mean(dim=1)  # (num_samples, num_features)
#                     uncert = concatenated.std(dim=1)  # (num_samples, num_features)
#                 else:
#                     agg = concatenated  # Bereits aggregiert
#                     uncert = torch.zeros_like(agg)
#                 self.live_visualizer.update(
#                     raw_importances=concatenated.detach().cpu().numpy(),
#                     aggregated_importance=agg.detach().cpu().numpy(),
#                     uncertainty=uncert.detach().cpu().numpy(),
#                     sample_idx=0
#                 )
#
#             # Live-Update: Ausgabe der kumulativen Importances
#             cumulative = torch.cat(all_importances, dim=0)
#             print(f"Nach Batch {batch_idx}: Kumulierte Importances-Shape: {cumulative.shape}")
#
#         if not all_importances:
#             raise ValueError("❌ No valid feature importances calculated!")
#         concatenated = torch.cat(all_importances, dim=0)
#         return concatenated if return_raw else concatenated
#
#     def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
#         mean_activation = np.mean(np.abs(activation), axis=0)
#         std_activation = np.std(np.abs(activation), axis=0)
#         normalized_mean = (mean_activation - np.min(mean_activation)) / (
#                     np.max(mean_activation) - np.min(mean_activation) + epsilon)
#         inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (
#                     np.max(std_activation) - np.min(std_activation) + epsilon)
#         aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
#         return aggregated_importance
#
#     def most_important_timestep(self, all_attributions):
#         return np.argmax(np.abs(all_attributions), axis=1)
#
#     def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
#         mean_activation = np.mean(np.abs(all_attributions), axis=1)
#         std_activation = np.std(np.abs(all_attributions), axis=1)
#         min_mean = np.min(mean_activation, axis=1, keepdims=True)
#         max_mean = np.max(mean_activation, axis=1, keepdims=True)
#         normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
#         min_std = np.min(std_activation, axis=1, keepdims=True)
#         max_std = np.max(std_activation, axis=1, keepdims=True)
#         normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
#         inverse_normalized_std = 1 - normalized_std
#         aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
#         return aggregated_importance
#
#     def aggregate_importances_over_samples(self, sample_importances):
#         global_importance = np.mean(sample_importances, axis=0)
#         return global_importance
#
#     def aggregate_and_return(self, sample_importances):
#         if isinstance(sample_importances, list):
#             sample_importances = np.vstack(sample_importances)
#         print("All Attributions:", sample_importances)
#         importance = self.calculate_actif_inverted_weighted_mean(sample_importances)
#         results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
#                    range(len(self.selected_features))]
#         print("Results from Analyzer:", results)
#         return results
#


# Beispiel: BayesianFeatureImportance als Singleton
class BayesianFeatureImportance(metaclass=Singleton):
    def __init__(self, model, features, device, use_gaussian_spread=True):
        """
        Initialize the analyzer.

        Args:
            model (torch.nn.Module): The trained model.
            features (list): List of feature names.
            device (str): The device to run the analysis on.
            use_gaussian_spread (bool): Flag to toggle between Gaussian spreading and equal distribution.
        """
        self.use_gaussian_spread = True
        self.max_indices = None
        self.layer_types = None
        self.activations = None
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.alpha = 0.1
        # Beispielhafte Initialisierung der Priorwerte (diese werden dann über alle Subjects hinweg aktualisiert)
        self.mu_prior = np.zeros(len(self.selected_features))
        self.sigma_prior = np.ones(len(self.selected_features))

        # Bayesian-Prior für jedes Feature (als Vektor, Länge = Anzahl Features)
        self.mu_prior = np.zeros(len(self.selected_features))
        self.sigma_prior = np.ones(len(self.selected_features))

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
                                importance[b, f] * torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
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
                layer_importance = torch.abs(activations)  # (batch, L, hidden)
            elif isinstance(layer, nn.Linear):
                activations = layer(activations)
                if activations.dim() == 2:
                    layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
                else:
                    layer_importance = torch.abs(activations)
            else:
                continue

            # Interpolation entlang der Feature-Dimension
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

            # Live-Update: Ausgabe der kumulativen Importances
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
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]
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

        # --- Bayesian Updating ---

    def update_bayesian(self, sample_importance, sigma_obs=0.1):
        """
        Aktualisiert die Priorverteilung (mu_prior, sigma_prior) für jedes Feature anhand
        einer neuen Beobachtung (sample_importance). Es wird angenommen, dass sowohl der Prior
        als auch die Beobachtung gaussisch verteilt sind.

        Args:
            sample_importance (np.ndarray): Vektor der Form (num_features,) mit den gemessenen Importances.
            sigma_obs (float): Beobachtungsstandardabweichung (fest oder vorgegeben).
        """
        for i in range(len(self.selected_features)):
            prior_var = self.sigma_prior[i] ** 2
            obs_var = sigma_obs ** 2
            posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
            posterior_mean = posterior_var * (self.mu_prior[i] / prior_var + sample_importance[i] / obs_var)
            self.mu_prior[i] = posterior_mean
            self.sigma_prior[i] = np.sqrt(posterior_var)
        # Debug-Ausgabe der aktualisierten Prior-Werte:
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
            device (str): Gerät, z.B. "cuda", "mps" oder "cpu".

        Returns:
            (np.ndarray, np.ndarray): Tuple mit den finalen mu_prior und sigma_prior (jeweils Vektoren der Länge num_features).
        """
        # Setze initiale Priorwerte (z.B. N(0,1) für alle Features)
        # self.mu_prior = np.zeros(len(self.selected_features))
        # self.sigma_prior = np.ones(len(self.selected_features))

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, dtype=torch.float32)
            # Aggregiere Importances pro Sample über die Zeit (ohne bayessches Update)
            sample_importances = self.forward_pass_with_importance(batch, return_raw=False)
            sample_importances_np = sample_importances.cpu().detach().numpy()  # (batch, num_features)
            for sample_imp in sample_importances_np:
                # Update the Bayesian priors for each sample
                self.update_bayesian(sample_imp, sigma_obs=0.1)
        print("Final Bayesian mu_prior:", self.mu_prior)
        print("Final Bayesian sigma_prior:", self.sigma_prior)
        return self.mu_prior, self.sigma_prior

    def compute_bayesACTIF(self, valid_loader):
        """
        Führt die bayessche DeepACTIF-Analyse durch: Es wird über den gesamten Datensatz
        iteriert, für jedes Sample wird die deterministisch aggregierte Importance berechnet
        und dann ein bayessches Update durchgeführt.

        Returns:
            dict: Dictionary mit 'mu' (Posterior-Mittelwerte) und 'sigma' (Posterior-Unsicherheiten) pro Feature.
        """
        # Wir führen den bayesschen Analyse-Durchlauf aus:
        mu_post, sigma_post = self.analyze_bayesian(valid_loader, self.device)
        results = [{'feature': self.selected_features[i],
                    'attribution': mu_post[i],
                    'uncertainty': sigma_post[i]} for i in range(len(self.selected_features))]
        print("Bayesian ACTIF Results:")
        for r in results:
            print(r)
        return results

# class BayesianFeatureImportance():
#     def __init__(self, model, features, device, use_gaussian_spread=True):
#         """
#         Initialize the analyzer.
#
#         Args:
#             model (torch.nn.Module): The trained model.
#             features (list): List of feature names.
#             device (str): The device to run the analysis on.
#             use_gaussian_spread (bool): Flag to toggle between Gaussian spreading and equal distribution.
#         """
#         self.model = model
#         self.selected_features = features if features is not None else []
#         self.device = device
#         self.use_gaussian_spread = use_gaussian_spread  # Flag to choose backprop method
#
#         self.activations = {}
#         self.layer_types = {}
#         self.max_indices = {}  # Store max-pooling indices
#         self.register_hooks()
#
#     def register_hooks(self):
#         """Registers forward hooks to store activations and max-pooling indices."""
#
#         def hook_fn(name, module):
#             def hook(module, input, output):
#                 if isinstance(output, tuple):  # Handle LSTM outputs
#                     output = output[0]
#                 self.activations[name] = output.detach()
#                 self.layer_types[name] = type(module).__name__
#                 # Store max-pooling indices correctly
#                 if isinstance(module, (nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                     _, max_idx = torch.max(output, dim=1)  # Shape (batch, features)
#                     self.max_indices[name] = max_idx
#
#             return hook
#
#         for name, layer in self.model.named_modules():
#             if isinstance(layer, (nn.Linear, nn.LSTM, nn.AdaptiveMaxPool1d, nn.MaxPool1d)):
#                 layer.register_forward_hook(hook_fn(name, layer))
#
#     def backpropagate_max_pooling(self, importance, activations, layer_name):
#         """
#         Backpropagates importance through the Max Pooling layer.
#
#         Args:
#             importance (Tensor): Shape (batch, features)
#             activations (Tensor): Shape (batch, timesteps, features)
#             layer_name (str): Name of the max pooling layer.
#
#         Returns:
#             Tensor: Expanded importance with shape (batch, timesteps, features)
#         """
#         print(f"🔄 Backpropagating through Max Pooling layer: {layer_name}")
#         if layer_name not in self.max_indices:
#             raise ValueError(
#                 f"🚨 No max_indices found for {layer_name}. Make sure max-pooling layers are hooked correctly!")
#         max_indices = self.max_indices[layer_name]  # (batch, features)
#         batch_size, timesteps, num_features = activations.shape
#         expanded_importance = torch.zeros_like(activations)
#         if self.use_gaussian_spread:
#             for b in range(batch_size):
#                 for f in range(num_features):
#                     max_t = max_indices[b, f].item()
#                     for t in range(timesteps):
#                         expanded_importance[b, t, f] = (
#                                 importance[b, f] * torch.exp(-0.5 * ((torch.tensor(t, dtype=torch.float32,
#                                                                                    device=importance.device) - max_t) / 2.0) ** 2)
#                         )
#         else:
#             for b in range(batch_size):
#                 for f in range(num_features):
#                     expanded_importance[b, max_indices[b, f], f] = importance[b, f]
#         print(f"✅ Final importance shape after Max Pooling: {expanded_importance.shape}")
#         return expanded_importance
#
#     def forward_pass_with_importance(self, inputs, return_raw=False):
#         """
#         Performs a forward pass and computes feature importances.
#
#         Args:
#             inputs (Tensor): Input tensor of shape (batch, timesteps, ...).
#             return_raw (bool): If True, returns the tensor with time information (shape: [batch, timesteps, num_features]),
#                                otherwise aggregates over the time axis (e.g. mean over time).
#
#         Returns:
#             Tensor: Raw importances (batch, timesteps, len(selected_features))
#                     or aggregated importances (batch, len(selected_features)).
#         """
#         self.model.eval()
#         activations = inputs
#         T = inputs.shape[1]
#         accumulated_importance = None
#
#         for name, layer in self.model.named_children():
#             if isinstance(layer, nn.LSTM):
#                 activations, _ = layer(activations)  # (batch, L, hidden)
#                 # For Bayesian method, we use sum over timesteps as in the original code
#                 layer_importance = torch.abs(activations).sum(dim=1)  # (batch, hidden)
#                 # Damit wir die Batch-Dimension behalten, formen wir um zu (batch, 1, hidden)
#                 layer_importance = layer_importance.unsqueeze(1)
#             elif isinstance(layer, nn.Linear):
#                 activations = layer(activations)  # Expected shape: (batch, features) or (batch, T, features)
#                 if activations.dim() == 2:
#                     layer_importance = torch.abs(activations).unsqueeze(1).expand(-1, T, -1)
#                 else:
#                     layer_importance = torch.abs(activations)
#             else:
#                 continue
#
#             # Interpolation along the feature dimension
#             N, T_current, orig_features = layer_importance.shape
#             reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
#             interpolated = F.interpolate(reshaped, size=len(self.selected_features),
#                                          mode="linear", align_corners=False)
#             interpolated_importance = interpolated.reshape(N, T_current, len(self.selected_features))
#             # Accumulate importances from different layers
#             if accumulated_importance is None:
#                 accumulated_importance = interpolated_importance
#             else:
#                 if accumulated_importance.shape != interpolated_importance.shape:
#                     raise ValueError(
#                         f"Shape mismatch: accumulated {accumulated_importance.shape} vs. current {interpolated_importance.shape}")
#                 accumulated_importance += interpolated_importance
#
#         if return_raw:
#             return accumulated_importance
#         else:
#             aggregated_importance = accumulated_importance.mean(dim=1)
#             return aggregated_importance
#
#     def analyze(self, dataloader, return_raw=False):
#         """
#         Computes feature importances over the entire dataset.
#
#         Args:
#             dataloader: DataLoader for the samples.
#             device (str): Device, e.g. "cuda" or "cpu".
#             return_raw (bool): If True, returns a tensor of shape (num_samples, timesteps, num_features).
#
#         Returns:
#             Tensor: If return_raw=False, returns an array of shape (num_samples, num_features)
#                     with aggregated values.
#         """
#         all_importances = []
#         for batch_idx, batch in enumerate(dataloader):
#             if isinstance(batch, (list, tuple)):
#                 batch = batch[0]
#             batch = batch.to(self.device, dtype=torch.float32)
#             importance = self.forward_pass_with_importance(batch, return_raw=return_raw)
#             if importance is None or torch.isnan(importance).any():
#                 print(f"⚠️ Skipping batch {batch_idx}, importance contains NaN!")
#                 continue
#             all_importances.append(importance)
#         if not all_importances:
#             raise ValueError("❌ No valid feature importances calculated!")
#         concatenated = torch.cat(all_importances, dim=0)
#         return concatenated if return_raw else concatenated
#
#     def calculate_actif_inverted_weighted_mean(self, activation, epsilon=1e-6):
#         """
#         Calculate feature importance using mean activations and inverse std deviation.
#
#         Args:
#             activation (np.ndarray): Array of shape (num_samples, num_features).
#             epsilon (float): Small constant for stabilization.
#
#         Returns:
#             np.ndarray: Adjusted importance (num_features,).
#         """
#         activation_abs = np.abs(activation)
#         mean_activation = np.mean(activation_abs, axis=0)
#         std_activation = np.std(activation_abs, axis=0)
#         normalized_mean = (mean_activation - np.min(mean_activation)) / (
#                     np.max(mean_activation) - np.min(mean_activation) + epsilon)
#         inverse_normalized_std = 1 - ((std_activation - np.min(std_activation)) / (
#                     np.max(std_activation) - np.min(std_activation) + epsilon))
#         adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
#         return adjusted_importance
#
#     def most_important_timestep(self, all_attributions):
#         """
#         Determines, for each sample and each feature, the timestep with the highest absolute value.
#
#         Args:
#             all_attributions (np.ndarray): Array of shape (num_samples, timesteps, num_features).
#
#         Returns:
#             np.ndarray: Array of shape (num_samples, num_features) with indices of the most important timestep.
#         """
#         mip = np.argmax(np.abs(all_attributions), axis=1)
#         return mip
#
#     def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
#         """
#         Aggregates raw importances over the time axis using the inverse weighted method.
#
#         Args:
#             all_attributions (np.ndarray): Array of shape (num_samples, timesteps, num_features).
#
#         Returns:
#             np.ndarray: Aggregated importances of shape (num_samples, num_features).
#         """
#         mean_activation = np.mean(np.abs(all_attributions), axis=1)
#         std_activation = np.std(np.abs(all_attributions), axis=1)
#         min_mean = np.min(mean_activation, axis=1, keepdims=True)
#         max_mean = np.max(mean_activation, axis=1, keepdims=True)
#         normalized_mean = (mean_activation - min_mean) / (max_mean - min_mean + epsilon)
#         min_std = np.min(std_activation, axis=1, keepdims=True)
#         max_std = np.max(std_activation, axis=1, keepdims=True)
#         normalized_std = (std_activation - min_std) / (max_std - min_std + epsilon)
#         inverse_normalized_std = 1 - normalized_std
#         aggregated_importance = (normalized_mean + inverse_normalized_std) / 2
#         return aggregated_importance
#
#     def aggregate_importances_over_samples(self, sample_importances):
#         """
#         Aggregates sample importances (which are assumed to be of shape (num_samples, num_features))
#         over all samples to produce a global importance per feature.
#
#         Args:
#             sample_importances (np.ndarray or torch.Tensor): Array/Tensor of shape (num_samples, num_features).
#
#         Returns:
#             pd.DataFrame: DataFrame with columns "feature" and "attribution".
#         """
#         if isinstance(sample_importances, torch.Tensor):
#             sample_importances = sample_importances.cpu().detach().numpy()
#         print("Input to aggregate_importances_over_samples, shape:", sample_importances.shape)
#         global_importance = np.mean(sample_importances, axis=0)  # (num_features,)
#         print("Global importance after aggregation:", global_importance.shape, global_importance)
#         if global_importance.shape[0] != len(self.selected_features):
#             raise ValueError(f"Expected {len(self.selected_features)} features, but got {global_importance.shape[0]}.")
#         df_final = pd.DataFrame({
#             "feature": self.selected_features,
#             "attribution": global_importance
#         })
#         print("Final DataFrame shape:", df_final.shape)
#         return df_final
#
#     def aggregate_and_return(self, sample_importances):
#         """
#         Aggregates the computed feature importances and formats them in SHAP style.
#
#         Args:
#             sample_importances (List[torch.Tensor]): List of feature importance vectors per batch.
#
#         Returns:
#             List[dict]: List of dictionaries per feature with keys 'feature' and 'attribution'.
#         """
#         if (isinstance(sample_importances, list) and len(sample_importances) == 0) or \
#                 (isinstance(sample_importances, torch.Tensor) and sample_importances.numel() == 0):
#             raise ValueError("❌ No feature importances calculated from the data.")
#         all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp
#                            for imp in sample_importances]
#         all_attributions = np.vstack(all_importances)
#         if np.isnan(all_attributions).any():
#             print("⚠️ Warning: NaN values detected in all_attributions!")
#             all_attributions = np.nan_to_num(all_attributions, nan=0.0)
#         print("All Attributions: ", all_attributions)
#         # Hier nehmen wir an, dass all_attributions bereits die Form (num_samples, num_features) hat.
#         importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
#         results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
#                    range(len(self.selected_features))]
#         print("Results from Class: ", results)
#         return results
#
#     def backpropagate_linear(self, importance, activations, layer_name):
#         """
#         Backpropagates importance through a Linear layer.
#         """
#         print(f"🔄 Backpropagating through Linear layer: {layer_name}")
#         if importance.dim() == 1:
#             importance = importance.unsqueeze(0)
#         if importance.dim() == 2:
#             importance = importance.unsqueeze(1)
#         target_size = (activations.shape[-1],)
#         print(f"⚠ Interpolating Linear layer from {importance.shape[-1]} to {target_size}")
#         importance = F.interpolate(importance, size=target_size, mode="linear", align_corners=False).squeeze(1)
#         print(f"✅ Backpropagated Linear Importance Shape: {importance.shape}")
#         return importance
#
#     def backpropagate_lstm(self, importance, activations, layer_name):
#         """
#         Backpropagates importance through an LSTM layer.
#         """
#         print(f"🔄 Backpropagating through LSTM layer: {layer_name}")
#         batch_size, timesteps, num_features = activations.shape
#         if importance.dim() == 2:
#             importance = importance.unsqueeze(1)
#         elif importance.dim() == 3 and importance.shape[1] == num_features:
#             importance = importance.permute(0, 2, 1)
#         print(f"✅ Adjusted importance shape before interpolation: {importance.shape}")
#         importance = importance.unsqueeze(1)
#         target_size = (timesteps, len(self.selected_features))
#         print(f"✅ Interpolating to match input shape {target_size}")
#         importance = F.interpolate(importance, size=target_size, mode="bilinear", align_corners=False).squeeze(1)
#         print(f"✅ Final importance shape after LSTM backpropagation: {importance.shape}")
#         return importance
#
#     def weighted_interpolation(self, importance, activations):
#         """
#         Spreads importance over timesteps using a Gaussian distribution centered at the max timestep.
#         """
#         batch_size, timesteps, num_features = activations.shape
#         max_indices = activations.abs().max(dim=1).indices
#         spread_importance = torch.zeros_like(activations)
#         for i in range(batch_size):
#             max_t = max_indices[i].item()
#             time_range = torch.arange(timesteps, device=importance.device)
#             gaussian_weights = torch.exp(-((time_range - max_t) ** 2) / (2 * (timesteps / 5) ** 2))
#             gaussian_weights = gaussian_weights / gaussian_weights.sum()
#             spread_importance[i] = gaussian_weights[:, None] * importance[i]
#         print(f"✅ Final importance shape after max-pooling backpropagation: {spread_importance.shape}")
#         return spread_importance

#
# class BayesianFeatureImportance:
#     def __init__(self, model, features=None, alpha=0.1):
#         self.model = model
#         self.selected_features = features if features is not None else []
#
#         # Initialize the prior for each feature: mean=0, std=1
#         self.mu_prior = np.zeros(len(self.selected_features))  # Prior mean
#         self.sigma_prior = np.ones(len(self.selected_features))  # Prior variance (uncertainty)
#
#         self.alpha = alpha  # Smoothing factor for EMA
#         self.device = next(model.parameters()).device  # Get the device of the model
#
#     def forward_pass_with_importance(self, inputs):
#         """
#         Passes inputs through each layer, collects activations, and interpolates importance back to input features.
#         """
#         self.model.eval()
#         activations = inputs
#         accumulated_importance = torch.zeros((inputs.shape[0], len(self.selected_features)), device=inputs.device)
#
#         for name, layer in self.model.named_children():
#             if isinstance(layer, nn.LSTM):
#                 activations, _ = layer(activations)
#                 layer_importance = activations.sum(dim=1)  # (batch, features)
#             elif isinstance(layer, nn.Linear):
#                 activations = layer(activations)
#                 layer_importance = activations.mean(dim=0)  # (features,)
#             else:
#                 continue  # Skip non-trainable layers
#
#             # Ensure layer_importance is at least 3D before interpolation
#             if layer_importance.dim() == 2:  # Shape (batch, features)
#                 layer_importance = layer_importance.unsqueeze(1)  # Convert to (batch, 1, features)
#             elif layer_importance.dim() == 1:  # Shape (features,)
#                 layer_importance = layer_importance.unsqueeze(0).unsqueeze(0)  # Convert to (1, 1, features)
#
#             # Interpolation back to the feature space
#             interpolated_importance = F.interpolate(
#                 layer_importance,  # Ensure it's (batch, 1, features)
#                 size=(len(self.selected_features),),  # Target feature count (must be a tuple)
#                 mode="linear",
#                 align_corners=False
#             ).squeeze(1)  # Remove the extra dimension after interpolation -> (batch, features)
#
#             # Accumulate importance
#             accumulated_importance += interpolated_importance
#
#         return accumulated_importance.mean(dim=0)  # Mean importance for the batch
#
#     def update_bayesian_distribution(self, sample_importance):
#         """
#         Update the Bayesian distribution (posterior) based on new feature importance.
#
#         Args:
#             sample_importance (torch.Tensor): Importance scores for the current sample.
#         """
#         # Update the posterior using Bayes' rule
#         sigma_obs = torch.std(sample_importance).item()  # Assume observed std dev is calculated
#         for i in range(len(self.selected_features)):
#             # Update the mean and variance of each feature
#             mu_prior = self.mu_prior[i]
#             sigma_prior = self.sigma_prior[i]
#             importance_i = sample_importance[i].item()
#
#             # Posterior mean and variance
#             posterior_mean = (sigma_obs ** 2 * mu_prior + sigma_prior ** 2 * importance_i) / (
#                         sigma_obs ** 2 + sigma_prior ** 2)
#             posterior_variance = (sigma_prior ** 2 * sigma_obs ** 2) / (sigma_prior ** 2 + sigma_obs ** 2)
#
#             # Update the feature's belief (posterior)
#             self.mu_prior[i] = posterior_mean
#             self.sigma_prior[i] = np.sqrt(posterior_variance)
#
#     def analyze(self, dataloader):
#         """
#         Computes feature importance over a dataset, updating the posterior distribution for each feature.
#         """
#         all_importances = []
#
#         for batch_idx, batch in enumerate(dataloader):
#             if isinstance(batch, (list, tuple)):
#                 batch = batch[0]  # Extract inputs if (inputs, labels) tuple
#
#             batch = batch.to(self.device, dtype=torch.float32)
#             feature_importance = self.forward_pass_with_importance(batch)
#
#             if feature_importance is None or torch.isnan(feature_importance).any():
#                 print(f"⚠️ Skipping batch {batch_idx}, feature importance contains NaN!")
#                 continue
#
#                 # Update Bayesian distribution with new feature importance
#             self.update_bayesian_distribution(feature_importance)
#
#             all_importances.append(feature_importance)
#
#         if not all_importances:
#             raise ValueError("❌ No valid feature importances calculated!")
#
#         # Convert tensors to NumPy
#         all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp for imp in
#                            all_importances]
#         all_attributions = np.vstack(all_importances)
#
#         # Debugging Check for NaNs
#         if np.isnan(all_attributions).any():
#             print("⚠️ Warning: NaN values detected in all_attributions!")
#             print("NaN Locations:", np.where(np.isnan(all_attributions)))
#             all_attributions = np.nan_to_num(all_attributions, nan=0.0)
#
#         # Apply ACTIF variant for aggregation
#         importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
#
#         # Handle NaNs in Importance Calculation
#         if np.isnan(importance).any():
#             print("⚠️ Replacing NaN values in final feature importance")
#             importance = np.nan_to_num(importance, nan=0.0)
#
#         results = [
#             [{'feature': self.selected_features[i], 'attribution': importance[sample_idx, i]}
#              for i in range(len(self.selected_features))]
#             for sample_idx in range(importance.shape[0])
#         ]
#
#         return results  # List of dicts per sample
#
#     def calculate_actif_inverted_weighted_mean(self, activation):
#         """
#         Calculate the importance by weighting high mean activations and low variability (stddev).
#
#         Args:
#             activation (np.ndarray): The input activations or features, shape (num_samples, num_features).
#
#         Returns:
#             adjusted_importance (np.ndarray): Adjusted importance where low variability is rewarded.
#         """
#         activation_abs = np.abs(activation)
#         mean_activation = np.mean(activation_abs, axis=0)
#         std_activation = np.std(activation_abs, axis=0)
#
#         # Normalize mean and invert normalized stddev
#         normalized_mean = (mean_activation - np.min(mean_activation)) / (
#                 np.max(mean_activation) - np.min(mean_activation))
#         inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (
#                 np.max(std_activation) - np.min(std_activation))
#
#         # Calculate importance as a combination of mean and inverse stddev
#         adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
#         return adjusted_importance

# Example usage:
# model = YourTrainedModel()
# analyzer = BayesianFeatureImportance(model, selected_features=['feature1', 'feature2', 'feature3'], device='cuda')

# all_results = analyzer.analyze(dataloader)
# print(all_results)

# class BayesianDeepACTIF_Analyzer:
#     def __init__(self, model, features, device):
#         self.model = model
#         self.selected_features = features if features is not None else []
#         self.device = device
#         self.activations = {}  # Store activations
#         self.means = {}  # Mean activations per layer
#         self.stds = {}  # Standard deviation of activations

#         # 🔹 Register hooks for Bayesian inference
#         self.register_hooks()

#     def register_hooks(self):
#         """Registers forward hooks to capture activations and compute likelihoods."""
#         def hook_fn(name):
#             def hook(module, input, output):
#                 if isinstance(output, tuple):  # Handle LSTM tuple outputs
#                     output = output[0]
#                 self.activations[name] = output.detach()

#                 # Compute mean and std for likelihood estimation
#                 self.means[name] = output.mean(dim=(0, 1))  # Mean over batch and time
#                 self.stds[name] = output.std(dim=(0, 1)) + 1e-6  # Avoid zero std
#             return hook

#         for name, layer in self.model.named_modules():
#             if isinstance(layer, (nn.Linear, nn.LSTM)):
#                 layer.register_forward_hook(hook_fn(name))

#     def bayesian_feature_importance(self, batch):
#         """
#         Performs Bayesian feature importance propagation through the model.
#         Uses prior (input layer), likelihood (hidden layers), and posterior (output).
#         """
#         batch_size, time_steps, num_features = batch.shape  # (batch, 10, 38)

#         # 🔹 Initialize prior (input layer)
#         prior_mu = torch.zeros(batch_size, num_features, device=batch.device)  # Shape: (batch, features)
#         prior_sigma = torch.ones(batch_size, num_features, device=batch.device)  # Shape: (batch, features)

#         activations = batch  # Initial input

#         # Store intermediate feature distributions
#         layer_mu = []
#         layer_sigma = []

#         for name, layer in self.model.named_children():
#             if isinstance(layer, nn.LSTM):
#                 activations, _ = layer(activations)  # Shape: (batch, time_steps, hidden_dim)

#                 # Compute likelihood statistics
#                 mu_L = activations.mean(dim=1)  # Aggregate over time steps -> Shape: (batch, hidden_dim)
#                 sigma_L = activations.std(dim=1)  # Standard deviation as uncertainty

#             elif isinstance(layer, nn.Linear):
#                 activations = layer(activations)  # Shape: (batch, features)
#                 mu_L = activations.mean(dim=0, keepdim=True)  # Shape: (1, features)
#                 sigma_L = activations.std(dim=0, keepdim=True)  # Shape: (1, features)

#             else:
#                 continue  # Skip non-trainable layers

#             # 🔹 Ensure prior_mu matches `mu_L` shape (expand prior over time if needed)
#             if mu_L.shape[1] != prior_mu.shape[1]:  # If features don't match
#                 prior_mu = prior_mu.unsqueeze(1).expand(-1, mu_L.shape[1])
#                 prior_sigma = prior_sigma.unsqueeze(1).expand(-1, mu_L.shape[1])

#             # 🔹 Compute Bayesian Posterior
#             posterior_mu = (prior_mu * sigma_L**2 + mu_L * prior_sigma**2) / (sigma_L**2 + prior_sigma**2)
#             posterior_sigma = (prior_sigma * sigma_L) / torch.sqrt(prior_sigma**2 + sigma_L**2)

#             # 🔹 Store posterior and move to next layer
#             layer_mu.append(posterior_mu)
#             layer_sigma.append(posterior_sigma)
#             prior_mu = posterior_mu
#             prior_sigma = posterior_sigma

#         print(f"✅ Final Bayesian Feature Importance Shape: {prior_mu.shape}")

#         return prior_mu.mean(dim=1)  # Aggregate -> Shape: (batch_size, final_features)

#     def analyze(self, dataloader):
#         """
#         Runs Bayesian DeepACTIF to compute feature importance per sample.
#         """
#         all_importances = []

#         for batch_idx, batch in enumerate(dataloader):
#             batch = batch[0].to(self.device)
#             sample_importance = self.bayesian_feature_importance(batch)

#             if sample_importance is not None:
#                 all_importances.append(sample_importance)
#             else:
#                 print(f"⚠️ Skipping batch {batch_idx}, feature importance is None!")

#         if not all_importances:
#             raise ValueError("❌ No feature importances calculated.")

#         all_importances = torch.cat(all_importances, dim=0)  # Shape: (num_samples, num_features)

#         print(f"✅ Final shape of all_attributions: {all_importances.shape}")
#         return all_importances  # Shape: (num_samples, num_features)

#     def aggregate_and_return(self, all_attributions):
#         """
#         Aggregates Bayesian feature importance and returns per-sample results in SHAP format.
#         """
#         all_importances = [imp.detach().cpu().numpy() if isinstance(imp, torch.Tensor) else imp for imp in all_attributions]
#         all_attributions = np.vstack(all_importances)

#         print(f"✅ Final shape of all_attributions: {all_attributions.shape}")

#         # Apply ACTIF aggregation
#         importance = self.calculate_actif_inverted_weighted_mean(all_attributions)
#         print(f"✅ Calculated importance shape: {importance.shape}")

#         if importance.shape[1] != len(self.selected_features):
#             raise ValueError(
#                 f"ACTIF method returned {importance.shape[1]} importance scores, "
#                 f"but {len(self.selected_features)} features were expected."
#             )

#         results = [
#             [{'feature': self.selected_features[i], 'attribution': importance[sample_idx, i]} 
#             for i in range(len(self.selected_features))]
#             for sample_idx in range(importance.shape[0])
#         ]

#         return results  # List of dicts per sample

#     def calculate_actif_inverted_weighted_mean(self, activation):
#         """
#         Calculate importance by combining mean activations and inverse std deviation.

#         Args:
#             activation (np.ndarray): Shape (num_samples, num_features).

#         Returns:
#             adjusted_importance (np.ndarray): Adjusted importance where low variability is rewarded.
#         """
#         activation_abs = np.abs(activation)
#         mean_activation = np.mean(activation_abs, axis=0)
#         std_activation = np.std(activation_abs, axis=0)

#         normalized_mean = (mean_activation - np.min(mean_activation)) / (np.max(mean_activation) - np.min(mean_activation))
#         inverse_normalized_std = 1 - (std_activation - np.min(std_activation)) / (np.max(std_activation) - np.min(std_activation))

#         adjusted_importance = (normalized_mean + inverse_normalized_std) / 2
#         return adjusted_importance
