# import libraries
import torch
from torch import nn
import argparse

class Encoder256(nn.Module):
    def __init__(self):
        super(Encoder256, self).__init__()
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

class Decoder256(nn.Module):
    def __init__(self):
        super(Decoder256, self).__init__()
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
            nn.ReLU(), # (None, 1, 256, 256)
        )
    
    def forward(self, x):
        y = self.decoder(x)
        return y

class Encoder1024(nn.Module):
    def __init__(self):
        super(Encoder1024, self).__init__()
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

class Decoder1024(nn.Module):
    def __init__(self):
        super(Decoder1024, self).__init__()
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

architectures = {256: (Encoder256(), Decoder256()), 1024: (Encoder1024(), Decoder1024())}

class ConvolutionalAE(nn.Module):
    def __init__(self, latent_size):
        super(ConvolutionalAE, self).__init__()
        # latent size
        self.latent_size = latent_size
        # Encoder-Decoder Architectures
        self.encoder, self.decoder = architectures[self.latent_size]
    
    def forward(self, x, return_latent=False):
        x = self.encoder(x)
        z = x.view(x.shape[0], -1)
        x = self.decoder(x)
        if return_latent:
            return x, z
        else:
            return x
    
    def test(self):
        _in_ = torch.randn(1, 1, 256, 256)
        _out_, z = self.forward(_in_, return_latent=True)
        assert _in_.shape == _out_.shape
        assert z.shape == (1, self.latent_size)
        print("Test Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent-size', default=256, type=int, help="latent size of the autoencoder")
    args = parser.parse_args()

    model = ConvolutionalAE(latent_size=args.latent_size)
    model.test()