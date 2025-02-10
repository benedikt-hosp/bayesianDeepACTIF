import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import memory_usage
import timeit
import logging
import matplotlib.pyplot as plt
import numpy as np


import threading
import queue
import time
import numpy as np
import matplotlib.pyplot as plt

class LiveVisualizer:
    def __init__(self, selected_features, update_interval=5):
        """
        Initialisiert den Visualizer.
        Args:
            selected_features (list): Liste der Feature-Namen.
            update_interval (int): Aktualisierungsintervall (z. B. alle n Iterationen).
        """
        self.selected_features = selected_features
        self.update_interval = update_interval

        # Erstelle einmalig die Figure und Achsen
        plt.ion()  # Interaktiver Modus
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle("Live Visualization of Feature Importances")

        # Linker Plot: Heatmap für raw Importances
        self.axs[0].set_xlabel("Features")
        self.axs[0].set_ylabel("Timesteps")
        self.axs[0].set_title("Raw Importances")
        self.im = self.axs[0].imshow(np.zeros((1, len(selected_features))), aspect='auto', cmap='viridis')
        self.fig.colorbar(self.im, ax=self.axs[0])

        # Rechter Plot: Balkendiagramm für aggregierte Importance
        self.axs[1].set_title("Aggregated Importance")
        self.axs[1].set_ylabel("Importance")
        self.bar_container = self.axs[1].bar(range(len(selected_features)), np.zeros(len(selected_features)))
        self.axs[1].set_xticks(range(len(selected_features)))
        self.axs[1].set_xticklabels(selected_features, rotation=45, ha='right')

        plt.show()

        # Queue für Daten, die vom Hauptthread an den Visualizer-Thread übergeben werden
        self.data_queue = queue.Queue()

        # Starte den Update-Thread als Daemon
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Endlosschleife, die die Queue abarbeitet und den Plot aktualisiert."""
        while True:
            try:
                # Warte auf neue Daten (Timeout z. B. 1 Sekunde, damit wir auch auf KeyboardInterrupt reagieren können)
                raw_importances, aggregated_importance, uncertainty, sample_idx, iteration = self.data_queue.get(timeout=1)
                self._update_plot(raw_importances, aggregated_importance, uncertainty, sample_idx, iteration)
            except queue.Empty:
                pass
            plt.pause(0.01)  # Kurze Pause, um den Plot-Eventloop laufen zu lassen

    def _update_plot(self, raw_importances, aggregated_importance, uncertainty, sample_idx, iteration):
        """Aktualisiert die beiden Subplots mit den neuen Daten."""
        # Aktualisiere linken Plot (Heatmap)
        self.axs[0].cla()
        im = self.axs[0].imshow(raw_importances[sample_idx], aspect='auto', cmap='viridis')
        self.axs[0].set_xlabel("Features")
        self.axs[0].set_ylabel("Timesteps")
        self.axs[0].set_title(f"Raw Importances (Sample {sample_idx})")
        self.fig.colorbar(im, ax=self.axs[0])

        # Aktualisiere rechten Plot (Balkendiagramm)
        self.axs[1].cla()
        indices = np.arange(len(self.selected_features))
        self.axs[1].bar(indices, aggregated_importance[sample_idx])
        self.axs[1].set_xticks(indices)
        self.axs[1].set_xticklabels(self.selected_features, rotation=45, ha="right")
        self.axs[1].set_title(f"Aggregated Importance (Sample {sample_idx})")
        self.axs[1].set_ylabel("Importance")

        # Optionale Anzeige von Unsicherheitswerten (z. B. als Fehlerbalken)
        # Hier könntest du z.B. self.axs[1].errorbar(...) aufrufen, wenn uncertainty vorliegt.

        self.fig.canvas.draw()

    def update(self, raw_importances, aggregated_importance, uncertainty, sample_idx=0, iteration=0):
        """
        Fügt neue Daten in die Queue ein, um den Plot zu aktualisieren.
        Aktualisierung erfolgt nur, wenn iteration % update_interval == 0.
        Args:
            raw_importances (np.ndarray): Array mit Form (num_samples, timesteps, num_features).
            aggregated_importance (np.ndarray): Array mit Form (num_samples, num_features).
            uncertainty (np.ndarray): Unsicherheitswerte (z.B. Standardabweichung) mit Form (num_samples, num_features).
            sample_idx (int): Welcher Sample angezeigt werden soll.
            iteration (int): Aktuelle Iteration (nur alle update_interval Iterationen wird aktualisiert).
        """
        if iteration % self.update_interval != 0:
            return
        # Neue Daten in die Queue einfügen
        self.data_queue.put((raw_importances, aggregated_importance, uncertainty, sample_idx, iteration))

#
# class LiveVisualizer:
#     def __init__(self, selected_features, update_interval=10):
#         self.selected_features = selected_features
#         self.update_interval = update_interval  # z. B. alle 5 Updates
#         plt.ion()  # Interaktiven Modus aktivieren
#         self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
#         self.fig.suptitle("Live Visualization of Feature Importances")
#         # Erstelle die ersten leeren Plots
#         # Linker Plot: Heatmap der raw Importances
#         self.im = self.axs[0].imshow(np.zeros((1, len(selected_features))), aspect='auto', cmap='viridis')
#         self.axs[0].set_xlabel("Features")
#         self.axs[0].set_ylabel("Timesteps")
#         self.axs[0].set_title("Raw Importances")
#         self.fig.colorbar(self.im, ax=self.axs[0])
#         # Rechter Plot: Balkendiagramm der aggregierten Werte (mit Unsicherheitsbalken)
#         self.bar_container = self.axs[1].bar(range(len(selected_features)), np.zeros(len(selected_features)))
#         self.axs[1].set_xticks(range(len(selected_features)))
#         self.axs[1].set_xticklabels(selected_features, rotation=45, ha='right')
#         self.axs[1].set_title("Aggregated Importance")
#         self.axs[1].set_ylabel("Importance")
#         plt.show()
#
#     def update(self, raw_importances, aggregated_importance, uncertainty, sample_idx=0, iteration=0):
#         """
#         Aktualisiert die Plots mit den neuen Werten.
#
#         Args:
#             raw_importances (np.ndarray): Array mit Form (num_samples, timesteps, num_features).
#             aggregated_importance (np.ndarray): Array mit Form (num_samples, num_features).
#             uncertainty (np.ndarray): Array mit Form (num_samples, num_features) – z. B. Standardabweichung.
#             sample_idx (int): Welcher Sample angezeigt werden soll.
#             iteration (int): Aktuelle Iteration; Aktualisierung erfolgt nur alle update_interval Iterationen.
#         """
#         # Aktualisiere nur, wenn iteration % update_interval == 0
#         if iteration % self.update_interval != 0:
#             return
#
#         # Linker Plot: Aktualisiere die Heatmap mit den raw Importances des ausgewählten Samples
#         self.axs[0].cla()
#         im = self.axs[0].imshow(raw_importances[sample_idx], aspect='auto', cmap='viridis')
#         self.axs[0].set_xlabel("Features")
#         self.axs[0].set_ylabel("Timesteps")
#         self.axs[0].set_title(f"Raw Importances (Sample {sample_idx})")
#         self.fig.colorbar(im, ax=self.axs[0])
#
#         # Rechter Plot: Aktualisiere das Balkendiagramm mit den aggregierten Werten und den Unsicherheitsbalken
#         self.axs[1].cla()
#         indices = np.arange(len(self.selected_features))
#         # Hier wird yerr verwendet, um die Unsicherheit darzustellen
#         self.axs[1].bar(indices, aggregated_importance[sample_idx],
#                         yerr=uncertainty[sample_idx], capsize=5)
#         self.axs[1].set_xticks(indices)
#         self.axs[1].set_xticklabels(self.selected_features, rotation=45, ha="right")
#         self.axs[1].set_title(f"Aggregated Importance (Sample {sample_idx})")
#         self.axs[1].set_ylabel("Importance")
#
#         plt.pause(0.001)
#         plt.draw()
