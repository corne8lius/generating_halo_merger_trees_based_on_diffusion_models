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

from check_consistency import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed_value=1238):
    "Set same seed to all random operations for reproduceability purposes"
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


#######################################################################################
#                           GAN
#######################################################################################


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
    

def create_generated_images(generator, latent_dim, reference_shape, noise_uniform = True):
    generator = generator.to(device)
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 64
    runs = math.floor(num_images / batch_size)
    extra = num_images % batch_size
    
    for run in range(runs + 1):
        if (run + 1) % 100 == 0:
            print(f"epochh {run} / {runs}")
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
    
    
    print(f"generated {images.shape[0]} images", flush=True)
    
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

    return distribution, nonzero_distribution


def get_large_sample_ks_significance_level(dist1, dist2, alpha):
    m = len(dist1.flatten())
    n = len(dist2.flatten())
    
    sig_level = np.sqrt(-np.log(alpha / 2) * ((1 + (m / n)) / (2 * m)))
    
    return sig_level

def ks_test(real_images, fake_images, dim = 0):
    print("Real images:", flush=True)
    real_dist, real_nonzero_dist = extract_distribution(real_images, dim)
    print("Generated images:", flush=True)
    fake_dist, fake_nonzero_dist = extract_distribution(fake_images, dim)

    statistic, p_value = ks_2samp(real_dist.cpu(), fake_dist.cpu())

    print("Full distribution:", flush=True)
    # Print the test results
    print("KS Statistic:", statistic, flush=True)
    print("P-value:", p_value, flush=True)

    alpha = 0.05  # significance level
    print(f"\nAccording to regular significance level of {alpha}:", flush=True)
    if p_value > alpha:
        print("The distributions are not significantly different (fail to reject H0)", flush=True)
    else:
        print("The distributions are significantly different (reject H0)", flush=True)
        
     # Interpret the results
    sig_level = get_large_sample_ks_significance_level(real_dist, fake_dist, alpha)  # significance level
    print(f"\nAccording to large sample significance level of {alpha}, giving significance level of {sig_level:.4f}:", flush=True)
    if statistic < sig_level:
        print(f"The distributions are not significantly different (fail to reject H0), KS statistic {statistic:.4f} < {sig_level:.4f}", flush=True)
    else:
        print(f"The distributions are significantly different (reject H0), KS statistic {statistic:.4f} > {sig_level:.4f}", flush=True)

    return statistic


def load_model(discriminator_path, encoder_path, generator_path,
               discriminator_model, encoder_model, generator_model):
    
    # load generator
    saved_generator = torch.load(generator_path, map_location='cpu')
    generator_trained = saved_generator["generator"]
    latent_dim = saved_generator["latent_space"]
    num_branches = saved_generator["num_branches"]
    nsnap = saved_generator["nsnap"]
    nvar = saved_generator["nvar"]
    
    # load encoder
    saved_encoder = torch.load(encoder_path, map_location='cpu')
    encoder_trained = saved_encoder["encoder"]
    
    # load discriminator
    saved_discriminator = torch.load(discriminator_path, map_location='cpu')
    discriminator_trained = saved_discriminator["discriminator"]

    # create generator
    generator = generator_model(nvar, num_branches, latent_dim)
    generator.load_state_dict(generator_trained)
                                
    # create encoder
    encoder = encoder_model(nvar, num_branches, latent_dim)
    encoder.load_state_dict(encoder_trained)
    
    # create discriminator
    discriminator = discriminator_model(nvar, num_branches)
    discriminator.load_state_dict(discriminator_trained)                        
                
                              
    return discriminator, encoder, generator, latent_dim

def plot_images(images, num_images):
    idx = torch.randint(0, images.shape[0] - 1, (num_images, 1))
    merger_trees = []
    for i in idx:
        print(f"image {i.item()}:")
        img = images[i].squeeze(0)
        merger_trees.append(img)
        img = img.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

def mass_not_preserved_percentage_decrease(dataset, threshold, original_value):
    branches = dataset.shape[-1]

    total_diffs = []

    not_preserving_mass = 0
    total_progenitors = 0

    max_decrease = threshold


    for im in dataset:
        for i in range(branches - 1):
            branch = im[:, i]
            nonzero_values = branch[branch.nonzero()].squeeze(-1)
            total_progenitors += len(nonzero_values)

            if len(nonzero_values) > 1:
                diffs = nonzero_values[1:] - nonzero_values[:-1]
                percentage_diffs = diffs / nonzero_values[:-1]
                not_preserved = percentage_diffs[percentage_diffs < max_decrease]
                not_preserving_mass += len(not_preserved)
                total_diffs.append(percentage_diffs.flatten().tolist())
                
    print(f"monotonicity threshold = {max_decrease}% change", flush=True)
    print(f"number of occurences where mass is not preserved = {not_preserving_mass}", flush=True)
    print(f"perc of occurences where mass is not preserved = {(100 * not_preserving_mass / total_progenitors):.2f}% vs. {original_value}% in training data", flush=True)
    print("\n", flush=True)
    return total_diffs

def distance_drastic_jumps(dataset, threshold):
    branches = dataset.shape[-1]

    total_diffs = []
    
    not_preserved_dist = 0
    total_progenitors = 0
    
    total_merger_distance_fail = 0


    for im in dataset:
        for i in range(branches):
            branch = im[:, i]
        
            nonzero_values = branch[branch.nonzero()].squeeze(-1)
            total_progenitors += len(nonzero_values)

            if len(nonzero_values) > 1:
                diffs = nonzero_values[1:] - nonzero_values[:-1]
                percentage_diffs = diffs / nonzero_values[:-1]
                not_preserved = percentage_diffs[percentage_diffs < 0.0]
                not_preserved_dist += len(not_preserved)
                total_diffs.append(percentage_diffs.flatten().tolist())
                
                minimum_branch = nonzero_values.min()
                last = nonzero_values[-1]
                if last != minimum_branch:
                    total_merger_distance_fail += 1
                
    print(f"total distance progentors = {total_progenitors} ", flush=True)
    print(f"number of occurences where distance increase (not preserved) = {not_preserved_dist}", flush=True)
    print(f"perc of occurences where distance increase (not preserved) = {(100 * not_preserved_dist / total_progenitors):.2f}% vs. 49.67% in training data", flush=True)
    print("\n", flush=True)
    print("total branches where the last halo distance to main branche is not the lowest in the branch is = ", total_merger_distance_fail, flush=True)
    perc = round(100 * total_merger_distance_fail / (branches * len(dataset)), 2)
    print(f"percentage of all branches where the last halo distance to main branch is not the lowest in the branch is = {perc}% vs. 17.88% in training data", flush=True)
    print("\n", flush=True)
    
    
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.tensor(jumps)

    abs_j = torch.abs(j)

    # Boolean indexing to get values above the threshold
    values_above_threshold = abs_j[abs_j > threshold]
    print(f"jumps greater than {100 * threshold}% (negative or positive) = {100 * (len(values_above_threshold) / (total_progenitors)):.2f}% of all jumps vs. 4.76% in training data", flush=True)

    return total_diffs, total_progenitors

def variable_consistancy_check(dataset):
    distance = dataset[:, 0, :, 1:]
    mass = dataset[:, 1]

    mass_thresholds = [-0.000001, -0.01, -0.05, -0.1]
    original_values_mass = [25.77, 11.12, 1.71, 0.32]
    distance_treshold = 0.2
    
    print("==================================================================================================================================", flush=True)
    print("MASS:", flush=True)
    print("==================================================================================================================================", flush=True)
    for i in range(len(mass_thresholds)):
        m_t = mass_thresholds[i]
        original_value = original_values_mass[i]
        total_diffs = mass_not_preserved_percentage_decrease(mass, m_t, original_value)
        print("\n")
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.nonzero(torch.tensor(jumps, dtype = torch.float32))
    if len(j) > 0:
        plt.hist(j.numpy(), bins = 100)
        plt.show()
        print(f"min mass change = {100 * j.min():.2f}% vs. -22.42% in training data", flush=True)
        print(f"max mass change = {100 * j.max():.2f}% vs. 28.61% in training data", flush=True)
        print(f"average mass change = {100 * torch.tensor(j, dtype = torch.float32).mean():.2f}% vs. 00.29% in training data", flush=True)
            
    
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)
    print("\n\n\n", flush=True)
    print("==================================================================================================================================", flush=True)
    print("DISTANCE:", flush=True)
    total_diffs, total_progenitors = distance_drastic_jumps(distance, distance_treshold)
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.nonzero(torch.tensor(jumps, dtype = torch.float32))
    if len(j) > 0:
        plt.hist(j.numpy(), bins = 100)
        plt.show()
        print(f"min jump = {100 * j.min():.2f} vs. -80.70% in training data", flush=True)
        print(f"max jump = {100 * j.max():.2f} vs. 509.33% in training data", flush=True)
        print(f"average jump = {100 * torch.tensor(j, dtype = torch.float32).mean():.2f} vs. -2.60% in training data", flush=True)
        print("==================================================================================================================================", flush=True)
        print("==================================================================================================================================", flush=True)



def analyze_data(generated_data_path, comparing_data_path):
    generated_data = torch.load(generated_data_path)
    original_data = torch.load(comparing_data_path)
    
    print("\nCheck structural consistency:", flush=True)
    consistant, inconsistant = check_consistency(generated_data)

    print("\nCheck variable consistency", flush=True)
    variable_consistancy_check(generated_data)
    
    print("\nCheck number of branches of all generated images:", flush=True)
    avg_branch = check_branch_length(generated_data)
    
    if len(consistant) != 0:
        print("\nCheck number of branches of consistent generated images:", flush=True)
        check_branch_length(consistant)
    
    print("\nMass KS test against training data:")
    ks = ks_test(original_data, generated_data, dim = 1)
    
    return generated_data

#######################################################################################
#                           Diffusion
#######################################################################################


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_timestep(model, x, t, betas):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    betas_t = get_index_from_list(betas, t, x.shape)
    
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
@torch.no_grad()
def generated_image(model, noise, T):
    img = noise
    betas = linear_beta_schedule(timesteps=T)
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, betas)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    return img


def create_generated_images_diffusion(model, img_size, reference_shape, T = 1000):
    model = model.to(device)
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 128
    runs = math.floor(num_images / batch_size)
    extra = num_images % batch_size
    transform = transforms.Compose([transforms.Resize((29, 10))])
    
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

        noise = torch.randn((batch_size, 3, img_size1, img_size2), device=device)
        
        generated = generated_image(model, noise, T)
            
        generated_images.append(generated)
    
    images = torch.cat(generated_images).detach()
    images = transform(images)
    
    #images[:, 2] = map_values(images[:, 2])
    
    print("reconstructed image:")
    i = random.randint(0, num_images - 1)
    img = images[i].permute(1, 2, 0).detach().cpu()
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.show()
    
    print(f"generated {images.shape[0]} images")
    
    return images

def custom_reconstruction_loss(output, target, scale = 10):
    # Calculate the reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction='none')
    
    # Apply higher penalty if target is zero
    penalty = torch.where((target == 0.0) |(target == 0.5) | (target == 1.0), scale * recon_loss, recon_loss)
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss


def train(dataloader, epochs, img_size, lr, device, name, model, diffusion_model):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = diffusion_model(T = T, img_size=img_size, device=device)
    l = len(dataloader)

    loss_list = []
    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader)
        loss_epoch = 0.0
        for i, images in enumerate(pbar):
            images = images.to(dtype=torch.float32)
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            x_t, noise = x_t.to(device), noise.to(device)
            predicted_noise = model(x_t, t).to(device)
            #loss = mse(noise, predicted_noise)
            loss = custom_reconstruction_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()

        loss_list.append(loss_epoch / len(dataloader))
        sampled_images = diffusion.sample(model, n=1)
        
        if epoch % 5 == 0:
            model_name = "notebooks/diffusion_notebook" + name
            torch.save(model.state_dict(), model_name)
            print("Saved model as: ", model_name)

        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
        
    return loss_list, sampled_images, model

def create_generated_images_diffusion2(model, img_size, reference_shape, T, interpolation, diffusion_model):
    model = model.to(device)
    diffusion = diffusion_model(T, img_size=img_size, device=device)
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

def unnormalize(tensor, mini = 0, maxi = 3.2445):
    
    return tensor * (maxi - mini) + mini
    
def de_log_normalization(tensor, in_scale = 1000, in_add = 1):
    
    return ((10 ** tensor) - in_add) / in_scale

def reset_normalize(tensor):
    return de_log_normalization(unnormalize(tensor))

def plot_graph_merger_tree_branch_mass_subhalo_type1(merger_tree):
    image = merger_tree.to(dtype=torch.float64)
    
    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = plt.cm.get_cmap("jet")

    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()

    new_indices = torch.stack([indices[:, 0], indices[:, 1], mass_nonzero, subh_nonzero], dim=0).permute(1, 0)
    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    # Plot the values
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            ax.plot(x[idx] + 1, y[idx], linestyle='-', color='black')

    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            ax.plot([x[i] + 1, 1], [y[i], y[i] + x[i]*0.05], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                ax.plot([x[i+1] + 1, 1], [y[i+1], y[i+1] + x[i+1]*0.05], color='black', linestyle='-')
            else:
                ax.plot([x[i] + 1, 1], [y[i+1], y[i+1] + x[i]*0.05], color='black', linestyle='-')

    ax.scatter(ind05[:, 1] + 1, ind05[:, 0], marker="^", s=50, c=ind05[:, 2], cmap=cmap, zorder=3)
    ax.scatter(ind1[:, 1] + 1, ind1[:, 0], marker="s", s=50, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('Branches')
    ax.set_ylabel('Snapshot')
    ax.set_ylim(29, 0)
    ax.set_title('Sample Plot with Reversed Y-axis')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value")

    return fig, ax

def plot_graph_merger_tree_dist_mass_subhalo_type1(merger_tree, merge=False):
    image = merger_tree.to(dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = plt.cm.get_cmap("jet")

    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()
    dist_nonzero = reset_normalize(dist[dist != 0.0].flatten())

    zero_positions = indices[:, 1] == 0.0

    new_dist = []
    dist_idx = 0
    for idx in zero_positions:
        if idx:
            new_dist.append(0.0)
        else:
            new_dist.append(dist_nonzero[dist_idx])
            dist_idx +=1

    new_indices = torch.stack([indices[:, 0], indices[:, 1],
                               mass_nonzero, subh_nonzero,
                               torch.Tensor(new_dist)],
                              dim=0).permute(1, 0)

    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    dist_nonzero = indices1[:, 4].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            ax.plot(dist_nonzero[idx], y[idx], linestyle='-', color='black')
            
    if merge:
        for i in range(len(x) - 1):
            if x[i] != x[i + 1]:
                ax.plot([dist_nonzero[i], 0], [y[i], y[i] + 1], color='black', linestyle='-')
            if i + 2 == len(x):
                val, count = np.unique(x, return_counts=True)
                if count[-1] == 1:
                    ax.plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')
                else:
                    ax.plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')


    ax.scatter(ind05[:, 4], ind05[:, 0], marker="^", s=50, c=ind05[:, 2], cmap=cmap, zorder=3)
    ax.scatter(ind1[:, 4], ind1[:, 0], marker="s", s=50, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('Distance from main branch')
    ax.set_ylabel('Snapshot')
    ax.set_ylim(29, 0)
    ax.set_title('Sample Plot with Reversed Y-axis')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value")
    
    
    return fig, ax

def plot_side_by_side(dataset, save = False, name = "test"):
    
    num_images = len(dataset)
    
    i = random.randint(0, num_images - 1)
    image = dataset[i]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 10))
    
    
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_xlabel('Branches', fontsize=20)
    axs[0].set_ylabel('Snapshot', fontsize=20)
    axs[0].set_title('Merger Tree image illustration', fontsize=20)
    
    cmap = plt.cm.get_cmap("jet")
    
    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()

    new_indices = torch.stack([indices[:, 0], indices[:, 1], mass_nonzero, subh_nonzero], dim=0).permute(1, 0)
    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    # Plot the values
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            axs[1].plot(x[idx] + 1, y[idx], linestyle='-', color='black')

    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            axs[1].plot([x[i] + 1, 1], [y[i], y[i] + x[i]*0.05], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                axs[1].plot([x[i+1] + 1, 1], [y[i+1], y[i+1] + x[i+1]*0.05], color='black', linestyle='-')
            else:
                axs[1].plot([x[i] + 1, 1], [y[i+1], y[i+1] + x[i]*0.05], color='black', linestyle='-')

    axs[1].scatter(ind05[:, 1] + 1, ind05[:, 0], marker="^", s=100, c=ind05[:, 2], cmap=cmap, zorder=3)
    axs[1].scatter(ind1[:, 1] + 1, ind1[:, 0], marker="s", s=75, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    axs[1].invert_yaxis()
    
    # Set labels and title
    axs[1].set_xlabel('Branches', fontsize=20)
    axs[1].set_ylabel('Snapshot', fontsize=20)
    axs[1].set_ylim(29, 0)
    axs[1].set_title("Merger Tree - branch", fontsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value", ax=axs[2])
    
    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()
    dist_nonzero = reset_normalize(dist[dist != 0.0].flatten())

    zero_positions = indices[:, 1] == 0.0

    new_dist = []
    dist_idx = 0
    for idx in zero_positions:
        if idx:
            new_dist.append(0.0)
        else:
            new_dist.append(dist_nonzero[dist_idx])
            dist_idx +=1

    new_indices = torch.stack([indices[:, 0], indices[:, 1],
                               mass_nonzero, subh_nonzero,
                               torch.Tensor(new_dist)],
                              dim=0).permute(1, 0)

    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    dist_nonzero = indices1[:, 4].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            axs[2].plot(dist_nonzero[idx], y[idx], linestyle='-', color='black')
            
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            axs[2].plot([dist_nonzero[i], 0], [y[i], y[i] + 1], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                axs[2].plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')
            else:
                axs[2].plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')


    axs[2].scatter(ind05[:, 4], ind05[:, 0], marker="^", label ='Sattelite', s=100, c=ind05[:, 2], cmap=cmap, zorder=3)
    axs[2].scatter(ind1[:, 4], ind1[:, 0], marker="s", label ='Main halo', s=75, c=ind1[:, 2], cmap=cmap, zorder=3)


    # Reverse the y-axis
    axs[2].invert_yaxis()
    
    # Set labels and title
    axs[2].set_xlabel('Distance from main branch', fontsize=20)
    axs[2].set_ylabel('Snapshot', fontsize=20)
    axs[2].set_ylim(29, 0)
    axs[2].set_title('Merger Tree - Distance to \n main branch', fontsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())

    
    plt.tight_layout()
    plt.legend(title='Markers', loc='upper right')
    if save:
        plt.savefig(f'merger_trees/{name}.png')
        print(f"saved fig {name}")
        
    plt.show()