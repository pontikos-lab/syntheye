# import packages
import torch
import torch.nn as nn

class multiClassPerceptron(nn.Module):
    """ Basic Multi-Class Logistic Regression Model """
    def __init__(self, in_channels, hidden_layers, out_channels):
        super(multiClassPerceptron, self).__init__()

        if len(hidden_layers) == 0:
            self.layers = nn.ModuleList([nn.Linear(in_channels, out_channels)])
        else:
            self.layers = [nn.Linear(in_channels, hidden_layers[0])]
            for i in range(len(hidden_layers) - 1):
                self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.Linear(hidden_layers[-1], out_channels))
            self.layers = nn.ModuleList(self.layers)

    def forward(self, input_):
        x = input_.view(len(input_), -1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x