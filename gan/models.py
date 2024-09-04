import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: (batch_size, 3, 64, 64)
            nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 128, 32, 32)
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 256, 16, 16)
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 512, 8, 8)
            
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (batch_size, 1024, 4, 4)
            
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1),
            # Output: (batch_size, 1, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.model = nn.Sequential(
            # Input: (batch_size, noise_dim, 1, 1)
            nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            # Output: (batch_size, 1024, 4, 4)
            
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Output: (batch_size, 512, 8, 8)
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: (batch_size, 256, 16, 16)
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (batch_size, 128, 32, 32)
            
            nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            # Output: (batch_size, output_channels, 64, 64)
        )

    def forward(self, x):
        x = x.view(-1, self.noise_dim, 1, 1)
        return self.model(x)
