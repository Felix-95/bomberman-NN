import torch.nn as nn
import torch.nn.functional as F

class MLPQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_actions, dropout_prob, activation_fn=nn.ReLU, train = True):
        """
        :param input_size: Number of input features
        :param hidden_size1: Number of neurons in the first hidden layer
        :param hidden_size2: Number of neurons in the second hidden layer
        :param n_actions: Number of possible actions (output size)
        :param dropout_prob: Dropout probability
        :param activation_fn: Activation function (default: ReLU)
        """
        super(MLPQNetwork, self).__init__()

        # If we are not training, we don't want to apply dropout
        if not train:
            dropout_prob = 0

        # Layers with dropout and simple activation function
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size2, n_actions)
        
        # Store the activation function (ReLU by default)
        self.activation_fn = activation_fn()
        
    def forward(self, x):
        """
        :param x: input tensor
        :return: vector of Q-values for each action
        """
        x = self.activation_fn(self.fc1(x))
        x = self.dropout1(x)
        
        x = self.activation_fn(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x