import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nvar, nbr):
        super().__init__()

        self.num_branches = nbr

        self.layers = nn.Sequential(
            nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        if self.num_branches == 6:
            self.linear = nn.Linear(11776, 1)
        else:
            self.linear = nn.Linear(35328, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return x


class Encoder(nn.Module):
    def __init__(self, nvar, nbr, latent_size):
        super().__init__()

        self.num_branches = nbr

        self.layers = nn.Sequential(
            nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.Flatten()
            )
        
        if self.num_branches == 6:
            self.linear = nn.Linear(11776, latent_size)
        else:
            self.linear = nn.Linear(35328, latent_size)

    def forward(self, x, t = 0):
        x = self.layers(x)
        x = self.linear(x)
        return x


class Generator(nn.Module):
    def __init__(self, nvar, nbr, latent_size):
        super().__init__()

        self.num_branches = nbr

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=1),
            nn.ELU(),
            nn.ConvTranspose2d(16, nvar, kernel_size=(1, 3), stride=1),
            #nn.Sigmoid()
            nn.ReLU()
        )
        
        if self.num_branches == 6:
            self.linear = nn.Linear(latent_size, 11776)
        else:
            self.linear = nn.Linear(latent_size, 35328)
        self.elu = nn.ELU()


    def forward(self, x, t = 0):
        x = self.elu(self.linear(x))
        if self.num_branches == 6:
            x = x.view(-1, 256, 23, 2)
        else:
            x = x.view(-1, 256, 23, 6)

        return self.layers(x)
        