import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


# Singleton-Metaklasse
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# Hilfsfunktion: Robust (MAD)-Schätzer; falls benötigt
def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med)) * 1.4826


class BayesianFeatureImportance(metaclass=Singleton):
    def __init__(self, model, features, device, use_gaussian_spread=True, initial_dataloader=None,
                 initial_alpha=0.1, decay_rate=0.01):
        """
        Initialisiert die bayessche Analyse. Optional wird ein kleiner Datensatz genutzt,
        um initial realistischere Priorwerte (μ und σ) zu berechnen.
        """
        self.model = model
        self.selected_features = features if features is not None else []
        self.device = device
        self.use_gaussian_spread = use_gaussian_spread

        # Lernratenparameter für das exponentiell gleitende Update
        self.initial_alpha = initial_alpha
        self.decay_rate = decay_rate
        self.update_count = 0

        # Initialisiere Priorwerte: Falls initial_dataloader übergeben, berechne robuste Schätzer;
        # ansonsten starte mit 0 bzw. 1.
        if initial_dataloader is not None:
            raw_importances = self.forward_pass_with_importance_from_dataloader(initial_dataloader, return_raw=True)
            # Berechne pro Feature den Median (als initiales μ) und MAD (als robusten Schätzer für σ)
            self.mu_prior = np.median(raw_importances, axis=(0, 1))
            # Hier: MAD über alle Samples und Timesteps (für jedes Feature)
            sigma_est = np.apply_along_axis(mad, 0, raw_importances.reshape(-1, raw_importances.shape[-1]))
            # Falls sigma_est an manchen Stellen zu 0 ist, setze einen kleinen Standardwert
            sigma_est[sigma_est == 0] = 0.1
            self.sigma_prior = sigma_est
        else:
            self.mu_prior = np.zeros(len(self.selected_features))
            self.sigma_prior = np.ones(len(self.selected_features))

        # Registriere ggf. Hooks, falls benötigt
        self.activations = {}
        self.layer_types = {}
        self.max_indices = {}
        self.register_hooks()

    def register_hooks(self):
        """Registriert Forward-Hooks in relevanten Schichten (z. B. LSTM, Linear)."""

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

    def forward_pass_with_importance(self, inputs, return_raw=False):
        """
        Führt einen Vorwärtsdurchlauf durch und berechnet dabei für jede Schicht
        die Absolutwerte der Aktivierungen, interpoliert diese auf die Anzahl der
        ausgewählten Features und akkumuliert sie.
        """
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

            N, T_current, orig_features = layer_importance.shape
            reshaped = layer_importance.reshape(N * T_current, 1, orig_features)
            interpolated = F.interpolate(reshaped, size=len(self.selected_features), mode="linear", align_corners=False)
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

    def forward_pass_with_importance_from_dataloader(self, dataloader, return_raw=False):
        """Hilfsmethode, um alle Importances aus einem Dataloader zu erhalten."""
        all_importances = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device, dtype=torch.float32)
            imp = self.forward_pass_with_importance(batch, return_raw=return_raw)
            all_importances.append(imp)
        all_importances = torch.cat(all_importances, dim=0)
        return all_importances

    def analyze(self, dataloader, return_raw=False):
        """Berechnet die Importances über den gesamten Datensatz."""
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
        """
        Berechnet den gewichteten Mittelwert, der hohe mittlere Aktivierungen und
        geringe Variabilität belohnt – über alle Samples hinweg.
        Hier wird über die Samples gemittelt (Achse 0).
        """
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
        """Ermittelt für jedes Sample und jedes Feature den Timestep mit maximalem Absolutwert."""
        return np.argmax(np.abs(all_attributions), axis=1)

    def aggregate_importance_inv_weighted(self, all_attributions, epsilon=1e-6):
        """
        Aggregiert raw Importances über die Zeit (Achse 1) mittels invers gewichteter Methode.
        """
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
        """
        Aggregiert die pro Sample berechneten Importances (Form: [num_samples, num_features])
        zu globalen Werten (Vektor der Länge num_features) und gibt ein DataFrame zurück.
        """
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
        """
        Aggregiert die berechneten Importances und formatiert sie im SHAP-Stil.
        """
        if isinstance(sample_importances, list):
            sample_importances = np.vstack(sample_importances)
        print("All Attributions:", sample_importances)
        importance = self.calculate_actif_inverted_weighted_mean(sample_importances)
        results = [{'feature': self.selected_features[i], 'attribution': importance[i]} for i in
                   range(len(self.selected_features))]
        print("Results from Analyzer:", results)
        return results

    # --- Bayesian Updating ---

    def update_bayesian(self, sample_importance, sigma_obs):
        """
        Aktualisiert die bayesschen Priorwerte (mu_prior und sigma_prior) für jedes Feature
        anhand einer neuen Beobachtung.

        Args:
            sample_importance (np.ndarray): Vektor der Form (num_features,) mit den aggregierten Importances.
            sigma_obs (np.ndarray): Vektor der Beobachtungsstandardabweichungen (σ_obs) pro Feature.
        """
        prior_var = self.sigma_prior ** 2
        obs_var = sigma_obs ** 2
        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
        posterior_mean = posterior_var * (self.mu_prior / prior_var + sample_importance / obs_var)
        # Dynamisch anpassende Lernrate (exponentiell gleitender Durchschnitt)
        current_alpha = self.initial_alpha / (1 + self.update_count * self.decay_rate)
        self.mu_prior = (1 - current_alpha) * self.mu_prior + current_alpha * posterior_mean
        self.sigma_prior = np.sqrt((1 - current_alpha) * (self.sigma_prior ** 2) + current_alpha * posterior_var)
        self.update_count += 1
        print("Updated Bayesian priors:")
        print("Mu:", self.mu_prior)
        print("Sigma:", self.sigma_prior)

    def analyze_bayesian(self, dataloader, device):
        """
        Führt einen vollständigen Durchlauf über den Datensatz durch und aktualisiert
        die bayesschen Priorwerte für jedes Feature anhand der pro Sample aggregierten Importances.
        Dabei wird für jedes Sample auch die Beobachtungsvarianz (σ_obs) aus den raw Importances
        (über die Timesteps) berechnet.

        Returns:
            (np.ndarray, np.ndarray): Tuple mit den finalen mu_prior und sigma_prior.
        """
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device, dtype=torch.float32)
            # Hole raw Importances (shape: [batch, timesteps, num_features])
            raw = self.forward_pass_with_importance(batch, return_raw=True)
            # Aggregiere pro Sample: Berechne Mittelwert und Standardabweichung über Timesteps (für jedes Feature)
            sample_imp = raw.mean(dim=1).cpu().detach().numpy()  # (batch, num_features)
            sigma_obs = raw.std(dim=1).cpu().detach().numpy()  # (batch, num_features)
            # Falls ein σ_obs-Wert in einem Feature zu klein ist, ersetze ihn durch einen Mindestwert (z. B. 0.01)
            sigma_obs[sigma_obs < 0.01] = 0.01
            for s in range(sample_imp.shape[0]):
                self.update_bayesian(sample_imp[s], sigma_obs[s])
            print(f"Nach Batch {batch_idx}: mu_prior: {self.mu_prior}, sigma_prior: {self.sigma_prior}")
        print("Final Bayesian mu_prior:", self.mu_prior)
        print("Final Bayesian sigma_prior:", self.sigma_prior)
        return self.mu_prior, self.sigma_prior

    def compute_bayesACTIF(self, valid_loader):
        """
        Führt die bayessche DeepACTIF-Analyse durch, indem für jedes Sample die
        aggregierte Importance berechnet und anschließend die bayesschen Priorwerte
        aktualisiert werden.

        Returns:
            List[dict]: Liste mit den finalen Posterior-Mittelwerten und Unsicherheiten (σ) pro Feature.
        """
        mu_post, sigma_post = self.analyze_bayesian(valid_loader, self.device)
        results = [{'feature': self.selected_features[i],
                    'attribution': mu_post[i],
                    'uncertainty': sigma_post[i]} for i in range(len(self.selected_features))]
        print("Bayesian ACTIF Results:")
        for r in results:
            print(r)
        return results

    # Weitere Methoden (backpropagate_linear, backpropagate_lstm, weighted_interpolation, etc.)
    # können hier wie in der Originalversion integriert bleiben.

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