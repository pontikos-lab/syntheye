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

class featureBased(nn.Module):
    def __init__(self):
        super(featureBased, self).__init__()
        self.encoder = self.load_autoencoder()
        # freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad=False
        self.dense = nn.Linear(in_features=2048, out_features=36)

    def load_autoencoder(self):
        # compress images into 1024 dim space using autoencoder
        from image_encoder.ae_model import ConvolutionalAE
        weights = torch.load("/home/zchayav/projects/syntheye/image_encoder/experiment_7/best_weights.pth")
        autoencoder = ConvolutionalAE(im_size=512, latent_size=2048)
        autoencoder.load_state_dict(weights)
        autoencoder.eval()
        return autoencoder.encoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.dense(x)
        return x

class simpleConvNet(nn.Module):
    """ Basic Multi-Class Logistic Regression Model """
    def __init__(self):
        super(simpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)

        self.dense1 = nn.Linear(in_features=64*64*32, out_features=2048)
        torch.nn.init.xavier_uniform_(self.dense1.weight)

        self.dense2 = nn.Linear(in_features=2048, out_features=36)
        torch.nn.init.xavier_uniform_(self.dense2.weight)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5) # TODO: Try p=0.2 and 0.1. 

    def forward(self, __input__):
        x = __input__ # (None, 1, 512, 512)

        # Conv network
        x = self.maxpool(self.relu(self.conv1(x))) # (None, 32, 256, 256)
        x = self.maxpool(self.relu(self.conv2(x))) # (None, 32, 128, 128)
        x = self.maxpool(self.relu(self.conv3(x))) # (None, 32, 64, 64)
        
        # FC network
        x = x.view(len(x), -1) # (None, 32*64*64)
        x = self.dropout(x) # (None, 32*64*64)
        x = self.relu(self.dense1(x)) # (None, 2048)
        out = self.dense2(x) # (None, 36)
        return out