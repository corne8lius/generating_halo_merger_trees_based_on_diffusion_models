import pandas as pd
import torch
import torch.nn as nn
import tqdm
import torch.nn.functional as F
import os
import numpy as np
import random
import re
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import math
import torchvision

class Discriminator_CGAN(nn.Module):
    def __init__(self, nvar, nsnap, nbr, printer):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(29 * 2 * 16, 1),
        )
        
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(29 * 2 * 16, 1)
        self.sigmoid = nn.Sigmoid()
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nDiscriminator:")
            print("input:", x.shape)
            x = self.elu(self.conv1(x))
            print("2:", x.shape)
            x = self.flatten(x)
            print("2:", x.shape)
            x = self.linear(x)
            print("3", x.shape)
            x = self.sigmoid(x)

            return x
    
        else:
            return self.layers(x)


class Encoder_CGAN(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(29 * 2 * 16, latent_size),
        )
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1)
        self.elu = nn.ELU()
        self.linear = nn.Linear(29 * 2 * 16, latent_size)
        self.flatten = nn.Flatten()
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\Encoder:")
            print("input:", x.shape)
            x = self.elu(self.conv1(x))
            print("1:", x.shape)
            x = self.flatten(x)
            print("2", x.shape)
            x = self.linear(x)
            print("3", x.shape)

            return x
        else:
            x = self.layers(x)
            """
            print("\nLatent space:")
            print("min:", x[0].min())
            print("max:", x[0].max())
            print("mean:", x[0].mean())
            """
            return x

class Generator_CGAN(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(16, nvar, kernel_size=(1, 5), stride=1),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(latent_size, 29 * 2 * 16)
        self.elu = nn.ELU()
        self.deconv1 = nn.ConvTranspose2d(16, nvar, kernel_size=(1, 5), stride=1)
        self.sigmoid = nn.Sigmoid() # since input is normalized to [0, 1]
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nGenerator:")
            print("input:", x.shape)
            x = self.elu(self.linear(x))
            print("1:", x.shape)
            x = x.view(-1, 16, 29, 2)
            print("2:", x.shape)
            x = self.sigmoid(self.deconv1(x))
            print("3", x.shape)
            return x
        else:
            x = self.elu(self.linear(x))
            x = x.view(-1, 16, 29, 2)

            return self.layers(x)
        
class Discriminator_big(nn.Module):
    def __init__(self, nvar, nsnap, nbr, printer):
        super().__init__()

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
            nn.Flatten(),
            nn.Linear(11776, 1)
        )
        
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(11776, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nDiscriminator:")
            print("0:", x.shape)
            x = self.elu(self.conv1(x))
            print("1:", x.shape)
            x = self.elu(self.conv2(x))
            print("2:", x.shape)
            x = self.elu(self.conv3(x))
            print("3:", x.shape)
            x = self.elu(self.conv4(x))
            print("4:", x.shape)
            x = self.elu(self.conv5(x))
            print("5:", x.shape)
            x = self.flatten(x)
            print("6:", x.shape)
            x = self.linear(x)
            x = self.sigmoid(x)

            return x
        else:
            
            return self.layers(x)


class Encoder_big(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

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
            nn.Flatten(),
            nn.Linear(11776, latent_size),
        )
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(11776, latent_size)
        
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nEncoder:")
            print("0:", x.shape)
            x = self.elu(self.conv1(x))
            print("1:", x.shape)
            x = self.elu(self.conv2(x))
            print("2:", x.shape)
            x = self.elu(self.conv3(x))
            print("3:", x.shape)
            x = self.elu(self.conv4(x))
            print("4:", x.shape)
            x = self.elu(self.conv5(x))
            print("5:", x.shape)
            x = self.flatten(x)
            print("6:", x.shape)
            x = self.linear(x)
            print("7:", x.shape)

            return x
        
        else:
            x = self.layers(x)
            return x


class Generator_big(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

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
            nn.ReLU(),
        )
        
        self.linear = nn.Linear(latent_size, 11776)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=1)
        self.deconv5 = nn.ConvTranspose2d(16, nvar, kernel_size=(1, 3), stride=1)
        
        self.printer = printer


    def forward(self, x):
        if self.printer:
            print("\nGenerator:")
            print("0:", x.shape)
            x = self.elu(self.linear(x))
            print("1:", x.shape)
            x = x.view(-1, 256, 23, 2)
            print("2:", x.shape)
            x = self.elu(self.deconv1(x))
            print("3:", x.shape)
            x = self.elu(self.deconv2(x))
            print("4:", x.shape)
            x = self.elu(self.deconv3(x))
            print("5:", x.shape)
            x = self.elu(self.deconv4(x))
            print("6:", x.shape)
            x = self.sigmoid(self.deconv5(x))
            print("7:", x.shape)

            return x
        else:
            x = self.elu(self.linear(x))
            x = x.view(-1, 256, 23, 2)

            return self.layers(x)
        
class Discriminator_big_full_data(nn.Module):
    def __init__(self, nvar, nsnap, nbr, printer):
        super().__init__()

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
            nn.Flatten(),
            nn.Linear(35328, 1)
        )
        
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(35328, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nDiscriminator:")
            print("0:", x.shape)
            x = self.elu(self.conv1(x))
            print("1:", x.shape)
            x = self.elu(self.conv2(x))
            print("2:", x.shape)
            x = self.elu(self.conv3(x))
            print("3:", x.shape)
            x = self.elu(self.conv4(x))
            print("4:", x.shape)
            x = self.elu(self.conv5(x))
            print("5:", x.shape)
            x = self.flatten(x)
            print("6:", x.shape)
            x = self.linear(x)
            x = self.sigmoid(x)

            return x
        else:
            
            return self.layers(x)


class Encoder_big_full_data(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

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
            nn.Flatten(),
            nn.Linear(35328, latent_size),
        )
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 3), stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 1), stride=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 1), stride=1)
        self.elu = nn.ELU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(35328, latent_size)
        
        self.printer = printer

    def forward(self, x):
        if self.printer:
            print("\nEncoder:")
            print("0:", x.shape)
            x = self.elu(self.conv1(x))
            print("1:", x.shape)
            x = self.elu(self.conv2(x))
            print("2:", x.shape)
            x = self.elu(self.conv3(x))
            print("3:", x.shape)
            x = self.elu(self.conv4(x))
            print("4:", x.shape)
            x = self.elu(self.conv5(x))
            print("5:", x.shape)
            x = self.flatten(x)
            print("6:", x.shape)
            x = self.linear(x)
            print("7:", x.shape)

            return x
        
        else:
            x = self.layers(x)
            return x


class Generator_big_full_data(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

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
            nn.ReLU(),
        )
        
        self.linear = nn.Linear(latent_size, 35328)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 1), stride=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 1), stride=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(3, 1), stride=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=(1, 3), stride=1)
        self.deconv5 = nn.ConvTranspose2d(16, nvar, kernel_size=(1, 3), stride=1)
        
        self.printer = printer


    def forward(self, x):
        if self.printer:
            print("\nGenerator:")
            print("0:", x.shape)
            x = self.elu(self.linear(x))
            print("1:", x.shape)
            x = x.view(-1, 256, 23, 6)
            print("2:", x.shape)
            x = self.elu(self.deconv1(x))
            print("3:", x.shape)
            x = self.elu(self.deconv2(x))
            print("4:", x.shape)
            x = self.elu(self.deconv3(x))
            print("5:", x.shape)
            x = self.elu(self.deconv4(x))
            print("6:", x.shape)
            x = self.sigmoid(self.deconv5(x))
            print("7:", x.shape)

            return x
        else:
            x = self.elu(self.linear(x))
            x = x.view(-1, 256, 23, 6)

            return self.layers(x)
        
        


def custom_reconstruction_loss(output, target, scale = 10):
    # Calculate the reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction='none')
    
    # Apply higher penalty if target is zero
    penalty = torch.where((target == 0.0) |(target == 0.5) | (target == 1.0), scale * recon_loss, recon_loss)
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss


def generator_classic_loss(disc_gen_values, epsilon = 1e-5):
    
    loss = 1 / len(disc_gen_values) * torch.sum(torch.log(disc_gen_values + epsilon))
    
    return loss



def discriminator_classic_loss(disc_real_values, disc_gen_values, epsilon = 1e-5):
    
    loss = 1 / len(disc_gen_values) * torch.sum(torch.log(disc_real_values + epsilon) + torch.log(1 - disc_gen_values + epsilon))
    
    return loss

def map_values(tensor, lower_threshold = 0.19, upper_threshold = 0.77):
    """
    Map values of a tensor between 0 and 1 to 0.0, 0.5, or 1.0 based on thresholds.

    Args:
    - tensor: Input tensor with values between 0 and 1.
    - lower_threshold: Lower threshold for mapping (inclusive).
    - upper_threshold: Upper threshold for mapping (exclusive).

    Returns:
    - mapped_tensor: Tensor with values mapped to 0.0, 0.5, or 1.0.
    """

    # Create a tensor of zeros with the same shape as the input tensor
    mapped_tensor = torch.zeros_like(tensor)

    # Map values based on thresholds
    mapped_tensor[[tensor <= lower_threshold]] = 0.0
    mapped_tensor[(tensor > lower_threshold) & (tensor < upper_threshold)] = 0.5
    mapped_tensor[tensor >= upper_threshold] = 1.0

    return mapped_tensor


def save_model(encoder, generator, discriminator,
               optimizer_discriminator, optimizer_enc_dec,
               disc_name, gen_name, enc_name,
               nbr, nsnap, nvar, latent_dim):
    state_dict_disc = {
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer_discriminator.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }


    state_dict_gen = {
                        "generator": generator.state_dict(),
                        "optimizer": optimizer_enc_dec.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }


    state_dict_enc = {
                        "encoder": encoder.state_dict(),
                        "optimizer": optimizer_enc_dec.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }

    torch.save(state_dict_disc, disc_name)
    print(f"saved discriminator as {disc_name}")
    
    torch.save(state_dict_gen, gen_name)
    print(f"saved generator as {gen_name}")
    
    torch.save(state_dict_enc, enc_name)
    print(f"saved encoder as {enc_name}")

def create_generated_images(generator, latent_dim, reference_shape, noise_uniform = True):
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 256
    runs = math.floor(num_images / 256)
    extra = num_images % 256
    
    for run in range(runs + 1):
        if run == runs:
            batch_size = extra
            
        if noise_uniform:
            high = 1.0
            low = -1.0
            uniform_noise = torch.rand(batch_size, latent_dim) * (high - (low)) + (low)
            noise = uniform_noise.to(device)
        
        else:
            noise = torch.randn(batch_size, latent_dim).to(device)
            
        fake_images = generator(noise)
        generated_images.append(fake_images)
    
    images = torch.cat(generated_images).detach()
    
    images[:, 2] = map_values(images[:, 2])
    
    print("reconstructed image:")
    i = random.randint(0, num_images - 1)
    img = images[i].permute(1, 2, 0).detach().numpy()
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.show()
    
    print(f"generated {images.shape[0]} images")
    
    return images
    
    


def extract_distribution(images, dim = 0):
    # extract variable
    variable = images[:, dim]
    
    # extract nonzero values
    nonzero_indices = torch.nonzero(variable.flatten())
    nonzero_value = variable.flatten()[nonzero_indices[:, 0]]
    
    # get nonzero distribution
    nonzero_distribution = nonzero_value
    
    # get regular distribution
    distribution = variable.flatten()
    
    # plot nonzero dist
    plt.hist(nonzero_value.flatten().numpy(), bins=100, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('distributions of Nonzero Values')
    plt.show()
    
    # plot regular dist
    plt.hist(variable.flatten().numpy(), bins=100, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('distributions of Nonzero Values')
    plt.show()
    print(f"number of nonzero values = {len(nonzero_value)}")
    
    return distribution, nonzero_distribution


    
def ks_test(real_images, fake_images, dim = 0):
    print("Real images:")
    real_dist, real_nonzero_dist = extract_distribution(real_images, dim)
    print("Generated images:")
    fake_dist, fake_nonzero_dist = extract_distribution(fake_images, dim)
    
    plt.hist(real_dist.numpy(), bins=100, alpha=0.5, label='real values', color='blue')
    plt.hist(fake_dist.numpy(), bins=100, alpha=0.5, label='generated values', color='orange')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Full distributions')
    plt.legend()

    # Show plot
    plt.show()
    
    
    statistic, p_value = ks_2samp(real_dist, fake_dist)

    print("Full distribution:")
    # Print the test results
    print("KS Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the results
    alpha = 0.05  # significance level
    if p_value > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
        
        
    
    plt.hist(real_nonzero_dist.numpy(), bins=100, alpha=0.5, label='real values', color='blue')
    plt.hist(fake_nonzero_dist.numpy(), bins=100, alpha=0.5, label='generated values', color='orange')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Nonzero distributions')
    plt.legend()

    # Show plot
    plt.show()
        
    statistic_nonzero, p_value_nonzero = ks_2samp(real_nonzero_dist, fake_nonzero_dist)
    print("Nonzero distribution:")
    # Print the test results
    print("KS Statistic:", statistic_nonzero)
    print("P-value:", p_value_nonzero)

    # Interpret the results
    alpha = 0.05  # significance level
    if p_value_nonzero > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
    

    return statistic, statistic_nonzero



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
        