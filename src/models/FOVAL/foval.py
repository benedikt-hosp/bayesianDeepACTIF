import torch.nn as nn
import numpy as np


class MaxOverTimePooling(nn.Module):
    def __init__(self):
        super(MaxOverTimePooling, self).__init__()

    def forward(self, x):
        return x.max(dim=1)[0]  # Max über die Zeitdimension (seq_length)


class Foval(nn.Module):
    def __init__(self, device, feature_count):
        super(Foval, self).__init__()

        self.maxpool = None
        self.device = device
        self.to(device)

        # Hyperparameteres
        self.hidden_layer_size = None
        self.feature_count = feature_count
        self.input_size = None
        self.embed_dim = None
        self.fc1_dim = None
        self.fc5_dim = None
        self.outputsize = 1
        self.dropout_rate = None

        # Layers
        self.input_linear = None
        self.lstm = None
        self.layernorm = None
        self.batchnorm = None
        self.fc1 = None
        self.fc5 = None
        self.activation = None
        self.dropout = None

        # Load Hyperparameteres from file

    def initialize(self, input_size, hidden_layer_size, fc1_dim, dropout_rate):

        # input_size = 34
        print("Hyperparameters of model: ", input_size, hidden_layer_size, fc1_dim, dropout_rate)
        # Linear layer to transform input features if needed
        self.input_linear = nn.Linear(in_features=input_size, out_features=input_size)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, num_layers=1, batch_first=True, hidden_size=hidden_layer_size)
        self.layernorm = nn.LayerNorm(hidden_layer_size)
        self.batchnorm = nn.BatchNorm1d(hidden_layer_size)
        self.maxpool = MaxOverTimePooling()  # Hier registrieren

        # Additional fully connected layers
        self.fc1 = nn.Linear(hidden_layer_size, fc1_dim // 4)  # Use integer division
        self.fc5 = nn.Linear(fc1_dim // 4, self.outputsize)  # Final FC layer for output
        self.activation = nn.ELU()

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(self.device)

    def forward(self, input_seq, return_intermediates=False):
        input_activations = self.input_linear(input_seq)

        # LSTM-Pass (hier verwendest du input_seq oder input_activations – je nach Bedarf)
        lstm_out, _ = self.lstm(input_seq)

        # Permutiere, damit die Dimensionen passen: (batch, hidden_layer_size, seq_length)
        lstm_out_perm = lstm_out.permute(0, 2, 1)

        # Wende Batch-Normalization an
        lstm_out_norm = self.batchnorm(lstm_out_perm)

        # Permutiere zurück zu (batch, seq_length, hidden_layer_size)
        lstm_out_3 = lstm_out_norm.permute(0, 2, 1)
        lstm_out_max = self.maxpool(lstm_out_3)  # Max über seq_length

        lstm_dropout = self.dropout(lstm_out_max)
        fc1_out = self.fc1(lstm_dropout)
        fc1_elu_out = self.activation(fc1_out)
        predictions = self.fc5(fc1_elu_out)

        if return_intermediates:
            intermediates = {'input_seq': input_seq,
                             'Input_activations': input_activations,
                             'Input_Weights': self.input_linear.weight.data.cpu().numpy(),
                             'LSTM_Out': lstm_out,
                             'LSTM_Weights_IH': self.lstm.weight_ih_l0.data.cpu().numpy(),
                             'LSTM_Weights_HH': self.lstm.weight_hh_l0.data.cpu().numpy(),
                             'Max_Timestep': lstm_out_max,
                             'FC1_Out': fc1_out,
                             'FC1_Weights': self.fc1.weight.data.cpu().numpy(),
                             'FC1_ELU_Out': fc1_elu_out,
                             'Output': predictions,
                             'FC5_Weights': self.fc5.weight.data.cpu().numpy()}
            return predictions, intermediates
        else:
            return predictions
