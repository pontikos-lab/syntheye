# import libraries
import torch
from torch import nn
from math import sqrt
import argparse

class Reshape(nn.Module):
    def __init__(self, in_features, out_channels):
        super(Reshape, self).__init__()
        self.in_features = in_features
        self.out_channels = out_channels
        self.k = 1 if self.in_features == self.out_channels else int(sqrt(self.in_features / self.out_channels))

    def forward(self, X):
        batch_size = X.shape[0]
        return X.view(batch_size, self.out_channels, self.k, self.k)

# =================================================

class Encoder_256_256(nn.Module):
    def __init__(self):
        super(Encoder_256_256, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 64, 64)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 16, 16)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 4, 4)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 2, 2)

            # block 8
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 2, 2)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 2, 2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 1, 1)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_256_256(nn.Module):
    def __init__(self):
        super(Decoder_256_256, self).__init__()
        self.decoder = nn.Sequential(
            # block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 2, 2)
            
            # block 2
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 4, 4)
            
            # block 3
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 8, 8)
            
            # block 4
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 16, 16)
            
            # block 5
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 32, 32)
            
            # block 6
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 64, 64)
            
            # block 7
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 128, 128)
            
            # block 8
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 256, 256)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder_256_1024(nn.Module):
    def __init__(self):
        super(Encoder_256_1024, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 64, 64)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 16, 16)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 4, 4)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 2, 2)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_1024_256(nn.Module):
    def __init__(self):
        super(Decoder_1024_256, self).__init__()
        self.decoder = nn.Sequential(

            # block 2
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 4, 4)
            
            # block 3
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 8, 8)
            
            # block 4
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 16, 16)
            
            # block 5
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 32, 32)
            
            # block 6
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 64, 64)
            
            # block 7
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 128, 128)
            
            # block 8
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 256, 256)

            # block 9 - final layer for grayscale conversion
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

# =================================================

class Encoder_512_512(nn.Module):
    def __init__(self):
        super(Encoder_512_512, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 256, 256)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 64, 64)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 16, 16)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 4, 4)

            # block 8
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 2, 2)

            # block 9
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # (None, 512, 2, 2)
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # (None, 512, 2, 2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 512, 1, 1)

            # flattening layer
            nn.Flatten() # (None, 512)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_512_512(nn.Module):
    def __init__(self):
        super(Decoder_512_512, self).__init__()
        self.decoder = nn.Sequential(
            # reshape layer
            Reshape(in_features=512, out_channels=512),

            # block 1
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(), # (None, 256, 2, 2)
            
            # block 2
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 4, 4)
            
            # block 3
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 8, 8)
            
            # block 4
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 16, 16)
            
            # block 5
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 32, 32)
            
            # block 6
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 64, 64)
            
            # block 7
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 128, 128)
            
            # block 8
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 256, 256)

            # block 9
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 512, 512)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder_512_1024(nn.Module):
    def __init__(self):
        super(Encoder_512_1024, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 256, 256)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 64, 64)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 16, 16)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 4, 4)

            # block 8
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 2, 2)

            # flattening layer
            nn.Flatten()
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_1024_512(nn.Module):
    def __init__(self):
        super(Decoder_1024_512, self).__init__()
        self.decoder = nn.Sequential(
            # reshape layer
            Reshape(in_features=1024, out_channels=256), # (None, 256, 2, 2)
            
            # block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 4, 4)
            
            # block 2
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 8, 8)
            
            # block 3
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 16, 16)
            
            # block 4
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 32, 32)
            
            # block 5
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 64, 64)
            
            # block 6
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 128, 128)
            
            # block 7
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 256, 256)

            # block 8
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 512, 512)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder_512_2048(nn.Module):
    def __init__(self):
        super(Encoder_512_2048, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 256, 256)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 64, 64)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 16, 16)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 4, 4)

            # flattening and linear layer
            nn.Flatten(), # (None, 4096)
            nn.Linear(in_features=4096, out_features=2048) # (None, 2048)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_2048_512(nn.Module):
    def __init__(self):
        super(Decoder_2048_512, self).__init__()
        self.decoder = nn.Sequential(
            # linear and reshape layer
            nn.Linear(in_features=2048, out_features=4096), # (None, 4096)
            Reshape(in_features=4096, out_channels=256), # (None, 256, 4, 4)
            
            # block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 8, 8)
            
            # block 2
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 16, 16)
            
            # block 3
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 32, 32)
            
            # block 4
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 64, 64)
            
            # block 5
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 128, 128)
            
            # block 6
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 256, 256)
            
            # block 7
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 512, 512)

            # block 8
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 512, 512)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder_512_4096(nn.Module):
    def __init__(self):
        super(Encoder_512_4096, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 256, 256)

            # block 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1), # (None, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 64, 64)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), # (None, 64, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 32, 32)

            # block 5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 16, 16)

            # block 6
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (None, 128, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 8, 8)
            
            # block 7
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # (None, 256, 8, 8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 4, 4)

            # flattening layer
            nn.Flatten() # (None, 4096)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_4096_512(nn.Module):
    def __init__(self):
        super(Decoder_4096_512, self).__init__()
        self.decoder = nn.Sequential(
            # reshape layer
            Reshape(in_features=4096, out_channels=256),
            
            # block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 8, 8)
            
            # block 3
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 16, 16)
            
            # block 4
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 32, 32)
            
            # block 5
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 64, 64)
            
            # block 6
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(), # (None, 8, 128, 128)
            
            # block 7
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(), # (None, 4, 256, 256)
            
            # block 8
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=2, stride=2),
            nn.BatchNorm2d(2),
            nn.ReLU(), # (None, 2, 512, 512)

            # block 9
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 512, 512)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder_512_65536(nn.Module):
    def __init__(self):
        super(Encoder_512_65536, self).__init__()
        self.encoder = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # (None, 16, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 16, 256, 256)

            # block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # (None, 16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 32, 128, 128)

            # block 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # (None, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 64, 64, 64)

            # block 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (None, 128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 128, 32, 32)

            # block 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (None, 256, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (None, 256, 16, 16)

            # flattening layer
            nn.Flatten() # (None, 256*16*16)
        )
    
    def forward(self, x):
        y = self.encoder(x)
        return y

class Decoder_65536_512(nn.Module):
    def __init__(self):
        super(Decoder_65536_512, self).__init__()
        self.decoder = nn.Sequential(
            # reshape layer
            Reshape(in_features=256*16*16, out_channels=256), # (None, 256, 16, 16)
            
            # block 1
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(), # (None, 128, 32, 32)
            
            # block 3
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(), # (None, 64, 64, 64)
            
            # block 4
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (None, 32, 128, 128)
            
            # block 5
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(), # (None, 16, 256, 256)
            
            # block 6
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
            nn.BatchNorm2d(1),
            nn.Tanh(), # (None, 1, 512, 512)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y


# =================================================

architectures = {256: {256: (Encoder_256_256(), Decoder_256_256()), 512: (), 1024: (Encoder_256_1024(), Decoder_1024_256()), 2048: ()},
                 512: {256: (), 512:(Encoder_512_512(), Decoder_512_512()), 1024: (Encoder_512_1024(), Decoder_1024_512()), 2048: (Encoder_512_2048(), Decoder_2048_512()), 4096: (Encoder_512_4096(), Decoder_4096_512()), 65536: (Encoder_512_65536(), Decoder_65536_512())}}

class ConvolutionalAE(nn.Module):
    def __init__(self, im_size, latent_size):
        super(ConvolutionalAE, self).__init__()
        # image dimension
        self.im_size = im_size
        # latent size
        self.latent_size = latent_size
        # Encoder-Decoder Architectures
        self.encoder, self.decoder = architectures[self.im_size][self.latent_size]
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def test(self):
        _in_ = torch.randn(1, 1, self.im_size, self.im_size)
        z = self.encoder(_in_)
        _out_ = self.decoder(z)
        assert _in_.shape == _out_.shape
        assert z.shape == (1, self.latent_size)
        print("Test Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-size', default=512, type=int, help="latent size of the autoencoder")
    parser.add_argument('--im-size', default=512, type=int, help="dimensions of image. Assumes a square shape image.")
    args = parser.parse_args()

    model = ConvolutionalAE(im_size=args.im_size, latent_size=args.latent_size)
    model.test()