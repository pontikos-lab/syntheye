# import packages
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

class simpleConvNet(nn.Module):
    """ Basic Multi-Class Logistic Regression Model """
    def __init__(self):
        super(simpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # self.conv3_bn = nn.BatchNorm2d(32)
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        # self.conv4_bn = nn.BatchNorm2d(64)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        # self.conv5_bn = nn.BatchNorm2d(64)
        self.dense1 = nn.Linear(in_features=246016, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=36)
        # self.dense3 = nn.Linear(in_features=64, out_features=36)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.0) # TODO: Try p=0.2 and 0.1. 

    def forward(self, __input__):
        x = __input__
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))
        # x = self.maxpool(self.relu(self.conv4(x)))
        # x = self.maxpool(self.relu(self.conv5(x)))
        x = x.view(len(x), -1)
        x = self.dropout(x)
        x = self.relu(self.dense1(x))
        # x = self.dropout(x) # TODO: Remove this dropout and try with only one dropout in the previous layer
        # x = self.relu(self.dense2(x))
        # x = self.dense3(x)
        out = self.dense2(x)
        return out