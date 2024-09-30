import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, n_actions, dropout_prob=0.25, activation_fn=F.leaky_relu, train = True): # relu6
        """
        :param input_size: Number of features
        :param hidden_size1: Number of neurons in the first hidden layer
        :param hidden_size2: Number of neurons in the second hidden layer
        :param n_actions: Number of possible actions
        :param dropout_prob: Dropout probability (optional)
        :param activation_fn: Activation function to use (optional) - relu6 is used by default, because it is the best for this task
        """
        super(MLPQNetwork, self).__init__()
        
        if not train:
            dropout_prob = 0

        # Define layers
        # We use batch normalization to speed up and stabilize training
        # We use drop out to prevent overfitting
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(hidden_size2, n_actions)
        
        # Store the activation function
        self.activation_fn = activation_fn
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        # Is used to initialize the weights of the model and make the model learn better
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        """
        :param x: input tensor
        :return: vector of Q-values, for each action
        """
        x = x.to(self.fc1.weight.device)
        # applying dropout before BatchNorm to prevent overfitting

        # Skip BatchNorm if batch size is 1
        if x.size(0) > 1:
            x = self.activation_fn(self.bn1(self.fc1(x)))
        else:
            x = self.activation_fn(self.fc1(x))
        
        x = self.dropout1(x)
         
        if x.size(0) > 1:
            x = self.activation_fn(self.bn2(self.fc2(x)))
        else:
            x = self.activation_fn(self.fc2(x))
        
        x = self.dropout2(x)
        x = self.fc3(x)
        return x