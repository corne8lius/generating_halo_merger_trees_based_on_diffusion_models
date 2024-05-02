import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
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
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#=============================================================================================================================================
                                    # Notebook 1
#=============================================================================================================================================

def read_folder(subfolder):
    folder_path = "../data/EAGLE100_matrixform/10branches/Original_trees/"
    path = folder_path + subfolder
    file_list = os.listdir(path)
    dfs = []
    idxs = []
    
    for file_name in file_list:
        file_path = os.path.join(path, file_name)
        
        match = re.search(r'\d+', file_name)
        if match:
            idx = torch.tensor(int(match.group()))
            idxs.append(idx)

        # Check if the file is a CSV file
        if file_name.endswith(".txt"):
            df = pd.read_csv(file_path, delim_whitespace=True, header=None)
            
            """
            print(file_name)
            print(type(df.values))
            print(df.values.shape)
            print(df.values)
            print(df)
            print(type(df.iloc[0]))
            print(df.iloc[0])
            """
            
            #tensor_df = torch.from_numpy(df.values)
            tensor_data = torch.tensor(df.values, dtype = torch.float64)
            dfs.append(tensor_data)

    paired_lists = zip(idxs, dfs)
    sorted_lists = sorted(paired_lists, key=lambda x: x[0])
    
    data = sorted_lists
    
    # Concatenate all DataFrames into a single DataFrame
    #data = torch.stack(sorted_lists)
    return data

#=============================================================================================================================================
                                    # Notebook 2
#=============================================================================================================================================


def normalize(dataset_change, minmax, channels):
    """
    input = [channels, dataset length, width, heigh]
    normalize given channels
    """
    num_chan = np.arange(dataset_change.shape[0])
    normalized_dataset = dataset_change
    for i in num_chan:
        if i in channels:  
            min_values = dataset_change[i].min()
            max_values = dataset_change[i].max()
            if minmax:
                print("minmax")
                normalized_dataset[i] = (dataset_change[i] - min_values) / (max_values - min_values)
            else:
                print("not minmax")
                normalized_dataset[i] = 2 * (dataset_change[i] - min_values) / (max_values - min_values) - 1
        else:
            print("not normalize")
            normalized_dataset[i] = dataset_change[i]
        
    return normalized_dataset

def one_hot_encode_tensor(tensor):
    """
    input: tensor of size (bs, 3 (channels), height, width)
    
    last channel should be subhalo
    
    output: tensor of size (bs, 5, height, width)
    
    """
    
    new_tensor = torch.clone(tensor[:, :2])

    subhalo_channel = tensor[:, 2]
    mapped_tensor = torch.zeros_like(subhalo_channel)

    mapped_tensor[[subhalo_channel == 0.0]] = 0
    mapped_tensor[[subhalo_channel == 0.5]] = 1
    mapped_tensor[[subhalo_channel == 1.0]] = 2

    one_hotted_test = torch.nn.functional.one_hot(mapped_tensor.to(torch.int64), 3)

    one_hotted = one_hotted_test.permute(0, 3, 1, 2)

    five_channel_tensor = torch.cat([new_tensor, one_hotted], dim = 1)
    
    return five_channel_tensor

def reverse_one_hot_tensor(tensor):
    
    new_tensor = torch.clone(tensor[:, :2])

    mapped_tensor = torch.zeros_like(tensor[:, 0])

    mapped_tensor[[tensor[:, 2] == 1.0]] = 0.0
    mapped_tensor[[tensor[:, 3] == 1.0]] = 0.5
    mapped_tensor[[tensor[:, 4] == 1.0]] = 1.0

    mapped_tensor = mapped_tensor.unsqueeze(1)

    reversed_one_hot = torch.cat([new_tensor, mapped_tensor], dim = 1)

    return reversed_one_hot

#=============================================================================================================================================
                                    # Notebook 3
#=============================================================================================================================================


def have_same_zero_values(tensor1, tensor2):
    # Create boolean masks for zero values
    zero_mask1 = (tensor1 == 0)
    zero_mask2 = (tensor2 == 0)
    
    # Compare the boolean masks
    print("the two tensors have zeroes in the same spots =", torch.all(zero_mask1 == zero_mask2).item())
    
    percentage = 100 * torch.sum(zero_mask1 == zero_mask2) / len(zero_mask1.flatten())
    
    print(f"the two tensors have zeroes and nonzero values in {percentage:.2f}% of the same spots")
    
    return


class Encoder_CGAN(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(29 * 6 * 16, latent_size),
        )
        
        self.conv1 = nn.Conv2d(nvar, 16, kernel_size=(1, 5), stride=1)
        self.elu = nn.ELU()
        self.linear = nn.Linear(29 * 6 * 16, latent_size)
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
            return self.layers(x)

class Generator_CGAN(nn.Module):
    def __init__(self, nvar, nsnap, nbr, latent_size, printer):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(16, nvar, kernel_size=(1, 5), stride=1),
            nn.ReLU()
        )
        
        self.linear = nn.Linear(latent_size, 29 * 6 * 16)
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
            x = x.view(-1, 16, 29, 6)
            print("2:", x.shape)
            x = self.sigmoid(self.deconv1(x))
            print("3", x.shape)
            return x
        else:
            x = self.elu(self.linear(x))
            x = x.view(-1, 16, 29, 6)

            return self.layers(x)
        
class AE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class Encoder_CGAN_big(nn.Module):
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
            nn.Linear(256 * 23 * 6, latent_size),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Generator_CGAN_big(nn.Module):
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
        
        self.linear = nn.Linear(latent_size, 256 * 23 * 6)
        self.elu = nn.ELU()

    def forward(self, x):

        x = self.elu(self.linear(x))
        x = x.view(-1, 256, 23, 6)

        return self.layers(x)
        

def training_notebook3_0(dataset, num_epochs, loss_function):
    nsnap = dataset.shape[2]
    nbr = dataset.shape[3]
    nvar = dataset.shape[1]
    printer = False

    batch_size = 128
    latent_size = 128

    loader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    encoder = Encoder_CGAN_big(nvar, nsnap, nbr, latent_size, printer).to(device)
    decoder = Generator_CGAN_big(nvar, nsnap, nbr, latent_size, printer).to(device)

    model = AE(encoder, decoder)

    loss_function = loss_function

    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

    epochs = num_epochs
    outputs = []
    losses = []
    images = []
    reconstructed_images = []
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        for image in tqdm.tqdm(loader):
            image = image.to(dtype=torch.float32)
            # Reshaping the image to (-1, 784)

            # Output of Autoencoder
            reconstructed = model(image)

            if epoch == epochs - 1:
                images.append(image)
                reconstructed_images.append(reconstructed.detach())

            # Calculating the loss function
            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())
            outputs.append((epochs, image, reconstructed))


        img = image[0].permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        img = reconstructed[0].permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        print("distance:")
        img = image[0, 0].unsqueeze(0).permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        img = reconstructed[0, 0].unsqueeze(0).permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses)
    
    return outputs, losses, images, reconstructed_images

def analyze_notebook3_0(reconstructed_images, images):
    reconstructed = torch.cat(reconstructed_images, dim = 0)
    images = torch.cat(images, dim = 0)

    reconstructed_dist = reconstructed[:, 0]
    image_dist = images[:, 0]

    print("\nOverall:")
    print("\nChecking zero values in reconstructed image:")
    print(f"minimum value in reconstructed image = {reconstructed.min():.10f}")
    print(f"Is minimum value in reconstructed image 0.0 = {reconstructed.min() == 0.0}")
    zero_count = (reconstructed == 0).sum().item()
    total_elements = reconstructed.numel()
    percentage_zero = (zero_count / total_elements) * 100
    print("Percentage of zero elements:", round(percentage_zero, 2), "%")

    print("\nChecking zero values in original image:")
    print(f"minimum value in reconstructed image = {images.min():.6f}")
    print(f"Is minimum value in reconstructed image 0.0 = {images.min() == 0.0}")
    zero_count = (images == 0).sum().item()
    total_elements = images.numel()
    percentage_zero = (zero_count / total_elements) * 100
    print("Percentage of zero elements:", round(percentage_zero, 2), "%")

    have_same_zero_values(images, reconstructed)


    print("\n")
    print("\n")
    print("\nDistance")
    print("\nChecking zero values in reconstructed image:")
    print(f"minimum value in reconstructed image = {reconstructed_dist.min():.10f}")
    print(f"Is minimum value in reconstructed image 0.0 = {reconstructed_dist.min() == 0.0}")
    zero_count = (reconstructed_dist == 0).sum().item()
    total_elements = reconstructed_dist.numel()
    percentage_zero = (zero_count / total_elements) * 100
    print("Percentage of zero elements:", round(percentage_zero, 2), "%")

    print("\nChecking zero values in original image:")
    print(f"minimum value in reconstructed image = {image_dist.min():.6f}")
    print(f"Is minimum value in reconstructed image 0.0 = {image_dist.min() == 0.0}")
    zero_count = (image_dist == 0).sum().item()
    total_elements = image_dist.numel()
    percentage_zero = (zero_count / total_elements) * 100
    print("Percentage of zero elements:", round(percentage_zero, 2), "%")

    have_same_zero_values(image_dist, reconstructed_dist)
#=============================================================================================================================================
                                    # Notebook 4
#=============================================================================================================================================

def similarity_percentage(tensor1, tensor2):
    # Check if tensors have the same shape
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape for comparison.")

    # Count the number of equal elements
    count_equal = torch.sum(tensor1 == tensor2).item()

    # Calculate the percentage similarity
    total_elements = tensor1.numel()
    percentage = (count_equal / total_elements) * 100.0

    return percentage

def have_same_zero_values(tensor1, tensor2):
    # Create boolean masks for zero values
    zero_mask1 = (tensor1 == 0)
    zero_mask2 = (tensor2 == 0)
    
    # Compare the boolean masks
    print("the two tensors have zeroes in the same spots =", torch.all(zero_mask1 == zero_mask2).item())
    
    percentage = 100 * torch.sum(zero_mask1 == zero_mask2) / len(zero_mask1.flatten())
    
    print(f"the two tensors have zeroes and nonzero values in {percentage:.2f}% of the same spots")
    
    return

def count_equality(images, reconstructed_images, lower_bound = 0.19, upper_bound = 0.77):
    
    images_stack = torch.cat(images, dim = 0)
    recon_images_stack = torch.cat(reconstructed_images, dim = 0)

    equal = 0
    sim = 0
    num = images_stack.shape[0]

    for i in range(num):
        img1 = images_stack[i, 2].unsqueeze(0)
        img2 = recon_images_stack[i, 2].unsqueeze(0)
        
        mapped = map_values(img2,lower_bound, upper_bound)

        if torch.equal(img1, mapped):
            equal += 1

        sim += similarity_percentage(img1, mapped)


    sim = sim / num
    
    print(f"Number of 100% equal subh images = {equal} / {num} = {(equal / num * 100):.2f}% of recon subh are equal original subh")
    print(f"Average similarity = {sim:.2f}%")

    return equal, round(sim, 4)

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

def show_values(tensor):
    unique_values, counts = torch.unique(tensor, return_counts=True)

    # Zip the unique values and counts together
    unique_counts = dict(zip(unique_values, counts))
    print(unique_counts)

    for i in unique_counts:
        print(f"value: {i}, count: {unique_counts[i]}")
        
    return

def custom_reconstruction_loss(output, target, target_threshold = 0, scale = 5):
    # Calculate the reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction='none')
    
    # Apply higher penalty if target is zero
    penalty = torch.where(target == target_threshold, scale * recon_loss, recon_loss)
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss

def reverse_one_hot_tensor(tensor):
    
    new_tensor = torch.clone(tensor[:, :2])

    mapped_tensor = torch.zeros_like(tensor[:, 0])

    mapped_tensor[[tensor[:, 2] == 1.0]] = 0.0
    mapped_tensor[[tensor[:, 3] == 1.0]] = 0.5
    mapped_tensor[[tensor[:, 4] == 1.0]] = 1.0

    mapped_tensor = mapped_tensor.unsqueeze(1)

    reversed_one_hot = torch.cat([new_tensor, mapped_tensor], dim = 1)

    return reversed_one_hot

def ternary_step(x, lower_threshold = 0.19, upper_threshold = 0.77):
    """
    Ternary step activation function.
    
    Args:
    - x: Input array or scalar
    - lower_threshold: Lower threshold for the activation function
    - upper_threshold: Upper threshold for the activation function
    
    Returns:
    - Output array where each element is set to:
      - 0 if the input is less than the lower threshold
      - 0.5 if the input is between the lower and upper thresholds
      - 1 if the input is greater than or equal to the upper threshold
    """
    return torch.where(x < lower_threshold, 0, torch.where(x < upper_threshold, 0.5, 1))

def equality(im, rec):
    equal = 0
    sim = 0
    num = im.shape[0]

    for i in range(num):
        img1 = im[i][2].unsqueeze(1)
        img2 = rec[i][2].unsqueeze(1)

        if torch.equal(img1, img2):
            equal += 1

        sim += similarity_percentage(img1, img2)


    sim = sim / num

    print(f"Number of 100% equal subh images = {equal} / {num} = {(equal / num * 100):.2f}% of recon subh are equal original subh")
    print(f"Average similarity = {sim:.2f}%")

def training(dataset, num_epochs, loss_function, five_channel = False, mapping = False, bounds = False):
    
    def reverse_one_hot_tensor(tensor):
    
        new_tensor = torch.clone(tensor[:, :2])

        mapped_tensor = torch.zeros_like(tensor[:, 0])

        mapped_tensor[[tensor[:, 2] == 1.0]] = 0.0
        mapped_tensor[[tensor[:, 3] == 1.0]] = 0.5
        mapped_tensor[[tensor[:, 4] == 1.0]] = 1.0

        mapped_tensor = mapped_tensor.unsqueeze(1)

        reversed_one_hot = torch.cat([new_tensor, mapped_tensor], dim = 1)

        return reversed_one_hot
    
    nsnap = dataset.shape[2]
    nbr = dataset.shape[3]
    nvar = dataset.shape[1]
    printer = False

    batch_size = 128
    latent_size = 300

    loader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
    encoder = Encoder_CGAN(nvar, nsnap, nbr, latent_size, printer).to(device)
    decoder = Generator_CGAN(nvar, nsnap, nbr, latent_size, printer).to(device)

    model = AE(encoder, decoder).to(device)

    loss_function = loss_function

    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

    epochs = num_epochs
    losses = []
    images2 = []
    reconstructed_images2 = []
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        reconstructed_images = []
        images = []
        for image in tqdm.tqdm(loader):
            optimizer.zero_grad()
            image = image.to(device)
            image = image.to(dtype=torch.float32)
            
            # Output of Autoencoder
            if bounds:
                reconstructed, lb, ub = model(image)
            
            elif mapping:
                reconstructed[:, 2] = map_values(reconstructed[:, 2])
            
            else:
                reconstructed = model(image)
                
                
            if epoch == epochs - 1 or epoch % 10 == 0:
                if five_channel:
                    image = reverse_one_hot_tensor(image)
                    reconstructed = reverse_one_hot_tensor(reconstructed)
                images.append(image)
                reconstructed_images.append(reconstructed.detach())
                
            if epoch == epochs - 1:
                images2.append(image)
                reconstructed_images2.append(reconstructed.detach())

            # Calculating the loss function

            loss = loss_function(reconstructed, image)

            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            loss.backward(retain_graph=True)
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.detach().numpy())

        if bounds:
            print("lower bound =", lb)
            print("upper bound =", ub)
            

        if epoch == epochs - 1 or epoch % 10 == 0:
            img = image[0].permute(1, 2, 0).detach().numpy()
            plt.imshow(img)
            plt.axis('off')  # Optional: Turn off axis ticks and labels
            plt.show()

            img = reconstructed[0].permute(1, 2, 0).detach().numpy()
            plt.imshow(img)
            plt.axis('off')  # Optional: Turn off axis ticks and labels
            plt.show()

            print("subhalo:")
            img = image[0, 2].unsqueeze(0).permute(1, 2, 0).detach().numpy()
            plt.imshow(img)
            plt.axis('off')  # Optional: Turn off axis ticks and labels
            plt.show()

            img = reconstructed[0, 2].unsqueeze(0).permute(1, 2, 0).detach().numpy()
            plt.imshow(img)
            plt.axis('off')  # Optional: Turn off axis ticks and labels
            plt.show()
            
            
            rec = torch.cat(reconstructed_images, dim = 0)
            im = torch.cat(images, dim = 0)

            have_same_zero_values(rec[:, 2], im[:, 2])

            zero_count = (rec[:, 2] == 0).sum().item()
            total_elements = rec[:, 2].numel()
            percentage_zero = (zero_count / total_elements) * 100
            print("Percentage of zero elements in reconstructed:", round(percentage_zero, 2), "%")

            zero_count = (im[:, 2] == 0).sum().item()
            total_elements = im[:, 2].numel()
            percentage_zero = (zero_count / total_elements) * 100
            print("Percentage of zero elements in original:", round(percentage_zero, 2), "%")

            reconstructed = torch.cat(reconstructed_images)
            subh = reconstructed[:, 2]

            nonzero_indices = torch.nonzero(subh.flatten())
            nonzero_value = subh.flatten()[nonzero_indices[:, 0]]

            # plot nonzero values
            plt.hist(torch.round(nonzero_value.flatten(), decimals = 4).numpy(), bins=100, color='blue')

            # Customize the plot
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title('Histogram of Tensor Values')

            # Show the plot
            plt.show()


    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses)
    
    return losses, images2, reconstructed_images2

def count_unique(tensor):
    tensor = torch.round(tensor, decimals = 2)
    unique_values, counts = tensor.unique(return_counts=True)
    return unique_values, counts

def custom_reconstruction_loss_subh(output, target, scale = 5):
    # Calculate the reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction='none')
    
    # Apply higher penalty if target is zero
    penalty = torch.where((target == 0.0) |(target == 0.5) | (target == 1.0), scale * recon_loss, recon_loss)
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss

def count_equality(t1, t2, lower_bound = 0.01, upper_bound = 0.75):

    equal = 0
    sim = 0
    num = t1.shape[0]

    for i in range(num):
        img1 = t1[i][2].unsqueeze(1)
        img2 = t2[i][2].unsqueeze(1)
        

        mapped = map_values(img2,lower_bound, upper_bound)

        if torch.equal(img1, mapped):
            equal += 1

        sim += similarity_percentage(img1, mapped)


    sim = sim / num
    
    print(f"Number of 100% equal subh images = {equal} / {num} = {(equal / num * 100):.2f}% of recon subh are equal original subh")
    print(f"Average similarity = {sim:.2f}%")

    return equal, round(sim, 4)
