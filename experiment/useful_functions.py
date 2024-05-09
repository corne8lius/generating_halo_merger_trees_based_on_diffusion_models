import torch
import tqdm
import torch.nn.functional as F
import os
import numpy as np
import random
import torchvision . transforms as T
import matplotlib.pyplot as plt
import torch.nn as nn
import math
from scipy.stats import ks_2samp
from torchvision import transforms
from torch import optim 

from experiment.diffusion_model import UNet, Diffusion
from experiment.evaluation import check_branch_length


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed_value=1238):
    """
    Set same seed to all random operations for reproduceability purposes
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


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

def create_generated_images_diffusion(model, img_size, reference_shape, T, interpolation, diffusion_model):
    """
    Generate merger trees given a diffusion model. 
    The number of merger trees that will be generated is the same as the reference_shape
    """
    model = model.to(device)
    diffusion = Diffusion(T = T, img_size=img_size, device=device)
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 16
    runs = math.floor(num_images / batch_size)
    extra = num_images % batch_size
    transform = transforms.Resize((29, 10), interpolation = interpolation)
    
    if type(img_size) == int:
        img_size1 = img_size
        img_size2 = img_size
    else:
        img_size1 = img_size[0]
        img_size2 = img_size[1]
    print(f"img size: ({img_size1}, {img_size2})")
    
    for run in range(runs + 1):
        if run % 10 == 0:
            print(f"epoch {run} / {runs}")
        if run == runs:
            batch_size = extra

        generated = diffusion.sample(model, batch_size).to(device)
            
        generated_images.append(generated)
    
    images = torch.cat(generated_images).detach()
    images = transform(images)
    
    return images

def normalize(dataset_change, minmax, channels):
    """
    normalize the given channels of a given dataset
    """
    num_chan = np.arange(dataset_change.shape[1])
    normalized_dataset = dataset_change
    for i in num_chan:
        if i in channels: 
            min_values = dataset_change[:, i].min().to(dtype=torch.float)
            max_values = dataset_change[:, i].max().to(dtype=torch.float)
            if minmax:
                normalized_dataset[:, i] = (dataset_change[:, i] - min_values).to(dtype=torch.float) / (max_values - min_values).to(dtype=torch.float)
            else:
                normalized_dataset[:, i] = 2 * (dataset_change[:, i] - min_values).to(dtype=torch.float) / (max_values - min_values).to(dtype=torch.float) - 1.0
        else:
            normalized_dataset[i] = dataset_change[i]
        
    return normalized_dataset

def replace_below_threshold(images, threshold, replacement_value):
    """
    Replace values in the input tensor that are below the threshold
    with the specified replacement value.
    
    Args:
    - tensor: input tensor
    - threshold: threshold value
    - replacement_value: value to replace with
    
    Returns:
    - tensor with replaced values
    """
    tensor = images.clone()
    # Create a mask for values below the threshold
    mask = tensor < threshold
    
    # Replace values below the threshold with the replacement value
    tensor[mask] = replacement_value
    
    return tensor

def transform_diffusion_image(images, d_thresh = 0.3, m_tresh = 0.5, s_low = 0.19, s_high = 0.77):
    """
    Takes a diffusion image and maps all values below thresholds of the given variables to zero

    for the subhalo variable, it maps to exactly 0.0, 0.5 and 1.0 given two thresholds
    """
    
    new_images = images.clone()
    
    new_images[:, 0] = replace_below_threshold(new_images[:, 0], d_thresh, 0.0)
    
    new_images[:, 1] = replace_below_threshold(new_images[:, 1], m_tresh, 0.0)
    
    new_images[:, 2] = map_values(new_images[:, 2], s_low, s_high)
    
    return new_images


def draw_sample_given_number_branch(dataset, num_branches):
    """
    Draw a random sample from a given dataset based on the desired number of branches 
    """

    total = check_branch_length(dataset, printer = False)
    merger_trees_given_branch_length = total[num_branches - 1]
    i = random.randint(0, len(merger_trees_given_branch_length))
    print(f"Sampling a generated merger tree with {num_branches} branches")
    print(f"\nPicked random sample number {i} out of {len(merger_trees_given_branch_length)} potential samples")
    sample = merger_trees_given_branch_length[i].unsqueeze(0)

    return sample

def draw_sample_given_complexity(dataset, higher_than, threshold):
    """
    Draw a random sample from a given dataset based on the desired complexity
    """
    attempts = len(dataset)
    if attempts > 10000:
        attempts = 10000
    for attempt in range(attempts):
        i = random.randint(0, len(dataset) - 1)

        nonzero_indices = torch.nonzero(dataset[i, 1].flatten())
        nonzero_value = len(dataset[i, 1].flatten()[nonzero_indices[:, 0]])

        if higher_than:
            if nonzero_value >= threshold:
                print(f"Generating merger tree with higher than {threshold} in complexity")
                print(f"\nPicked random sample number {i} out of {len(dataset)} potential samples")
                sample = dataset[i]
                check_branch_length(sample.unsqueeze(0))
                return sample
            else:
                continue

        elif not higher_than:
            if nonzero_value <= threshold:
                print(f"Generating merger tree with less than {threshold} in complexity")
                print(f"\nPicked random sample number {i} out of {len(dataset)} potential samples")
                sample = dataset[i]
                check_branch_length(sample.unsqueeze(0), True, False)
                return sample
            else:
                continue
        
    print("Could not find a tree with the desired complexity")

    print("Generating random sample:")
    sample = dataset[i]
    return sample
    

def draw_sample_given_branch_and_complexity(dataset, num_branches = None, threshold = None, higher_than = False):
    """
    Draw a random sample from a given dataset based on the desired number of branches and the complexity
    """

    if num_branches is None and threshold is None:
        print("You must specify either a given number of branches or a threshold, generating random merger tree without any specifications:")
        i = random.randint(0, len(dataset))
        sample = dataset[i]
        return sample.unsqueeze(0)

    elif num_branches is not None:
        print(f"Sampling a generated merger tree with {num_branches} branches")

        total = check_branch_length(dataset, printer = False, print_full = False)
        merger_trees_given_branch_length = total[num_branches - 1]

        if threshold is not None:
            trees = torch.stack(merger_trees_given_branch_length)
            
            sample = draw_sample_given_complexity(trees, higher_than, threshold)

            return sample.unsqueeze(0)
        
        elif threshold is None:
            i = random.randint(0, len(merger_trees_given_branch_length))
            sample = merger_trees_given_branch_length[i]
            return sample.unsqueeze(0)
        
    elif threshold is not None:
        sample = draw_sample_given_complexity(dataset, higher_than, threshold)
        return sample.unsqueeze(0)