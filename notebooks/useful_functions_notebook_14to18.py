import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch import nn
import math
import tqdm
import torchvision
import numpy as np
import logging
from scipy.stats import ks_2samp
import random
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Resize((29, 10)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    
def show_tensor_image_no_resize(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
    
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    
    model.to(device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred), x_noisy

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
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
def sample_plot_image(model, T, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    # Sample noise
    img_size = 64
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    plt.show()
    
@torch.no_grad()
def sample_plot_image_no_resize(model, T, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance):
    # Sample noise
    img_size = 64
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image_no_resize(img.detach().cpu())
    plt.show()

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


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm.tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cpu"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
            t = t.unsqueeze(-1).type(torch.float)
            t = self.pos_encoding(t, self.time_dim)

            x1 = self.inc(x)
            x2 = self.down1(x1, t)
            x2 = self.sa1(x2)
            x3 = self.down2(x2, t)
            x3 = self.sa2(x3)
            x4 = self.down3(x3, t)
            x4 = self.sa3(x4)

            x4 = self.bot1(x4)
            x4 = self.bot2(x4)
            x4 = self.bot3(x4)
            
            x = self.up1(x4, x3, t)
            x = self.sa4(x)
            x = self.up2(x, x2, t)
            x = self.sa5(x)
            x = self.up3(x, x1, t)
            x = self.sa6(x)
            output = self.outc(x)
            return output

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Resize((29, 10)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def normalize(dataset_change, minmax, channels):
    """
    input = [channels, dataset length, width, heigh]
    normalize given channels
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



def check_thresholds(tensor, lower_threshold, upper_threshold):
    # Count the number of values between the thresholds
    count_between_thresholds = torch.sum((tensor > lower_threshold) & (tensor < upper_threshold))

    # Count the total number of values in the tensor
    total_values = tensor.numel()

    # Calculate the percentage
    percentage_between_thresholds = (count_between_thresholds / total_values) * 100

    print("Percentage of values between the thresholds:", percentage_between_thresholds.item(), "%")



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
    
    new_images = images.clone()
    
    new_images[:, 0] = replace_below_threshold(new_images[:, 0], d_thresh, 0.0)
    
    new_images[:, 1] = replace_below_threshold(new_images[:, 1], m_tresh, 0.0)
    
    new_images[:, 2] = map_values(new_images[:, 2], s_low, s_high)
    
    return new_images

def check_branch_length(dataset):
    one_branch = []
    two_branch = []
    three_branch = []
    four_branch = []
    five_branch = []
    six_branch = []
    seven_branch = []
    eight_branch = []
    nine_branch = []
    ten_branch = []

    for datapoint in dataset:
        one_channel = datapoint[1]
        branches = torch.count_nonzero(one_channel, dim=0)
        num_branch = torch.count_nonzero(branches, dim=0)
        if num_branch == 1:
            one_branch.append(datapoint)
        elif num_branch == 2:
            two_branch.append(datapoint)
        elif num_branch == 3:
            three_branch.append(datapoint)
        elif num_branch == 4:
            four_branch.append(datapoint)
        elif num_branch == 5:
            five_branch.append(datapoint)
        elif num_branch == 6:
            six_branch.append(datapoint)
        elif num_branch == 7:
            seven_branch.append(datapoint)
        elif num_branch == 8:
            eight_branch.append(datapoint)
        elif num_branch == 9:
            nine_branch.append(datapoint)
        elif num_branch == 10:
            ten_branch.append(datapoint)

    total = [one_branch, two_branch, three_branch, four_branch, five_branch,
             six_branch, seven_branch, eight_branch, nine_branch, ten_branch]

    total_im = 0
    for i, branch_list in enumerate(total):
        total_im += len(branch_list)
        print(f"number of images with {i + 1} branches is: {len(branch_list)}")

    print(f"double check that all images are counted: total images is {total_im} = {dataset.shape[0]}")
    
def check_zeros_same_index(image):
    dist = image[:, 0]
    mass = image[:, 1]
    subh = image[:, 2]

    same_zero_indices = (dist == 0) & (mass == 0) & (subh == 0)

    equal = same_zero_indices.sum() / (subh == 0).sum()
    
    return equal

def check_zero_dist_main_branch(image):
    tree = image[0].unsqueeze(0)
    dist = tree[:, 0]
    main_branch = dist[:, :, 0]
    main_branch_zero = sum(main_branch == 0).sum() / len(main_branch.flatten())
    
    return main_branch_zero

def check_gaps_between_branches(image):
    where = 0
    where_chan = 0
    inconsistent = False
    for channel in range(image.shape[1]):
        stop_branch = False
        for bran in range(image.shape[-1]):
            if channel == 0 and bran == 0:
                continue
            branch = image[:, channel,: ,bran]
            if branch.sum() > 0:
                if stop_branch:
                    where = bran
                    where_chan = channel
                    inconsistent = True
                    break
                else:
                    continue
            elif branch.sum() == 0:
                stop_branch = True

        if inconsistent:
            break
            
    return inconsistent, where, where_chan


def check_gaps_in_branch(image):
    where_bran = 0
    where_row = 0
    inconsistent = False
    for channel in range(image.shape[1]):
        for bran in range(image.shape[-1]):
            stop_branch = False
            branch_started = False
            for row in range(image.shape[2]):
                pixel = image[:, channel, row ,bran]
                if pixel == 0 and branch_started == False:
                    continue
                elif pixel == 0 and branch_started:
                    stop_branch = True
                    continue
                elif pixel != 0 and stop_branch == False:
                    branch_started = True
                    continue
                elif pixel != 0 and stop_branch:
                    where_bran = bran
                    where_row = row
                    inconsistent = True
                    break
            if inconsistent:
                break
                
        if inconsistent:
            break
            
    return inconsistent, where_bran, where_row

def check_last_descendant(image):
    mass = image[0, 1, -1, 0] != 0.0
    subh = image[0, 2, -1, 0] != 0.0
    
    exist = mass and subh

    only_last = (sum(image[0, 1, -1, :]) == image[0, 1, -1, 0]) and (sum(image[0, 2, -1, :]) == image[0, 2, -1, 0])

    exist_and_only = exist and only_last
    
    return exist_and_only
    
    
    
    
def check_consistency(images, print_index = False):
    
    consistant_images = []
    inconsistent_images = []
    num_images = images.shape[0]
    
    if print_index:
        print("inconsistent images:")
    
    zero_inconsistency = 0
    dist_main_branch = 0
    gap_between_branch = 0
    gap_in_branch = 0
    last_descendant = 0
    
    two_or_more_inconsistencies = 0
    
    for i, image in enumerate(images):
        image = image.unsqueeze(0)
        # check zeros in same index
        zero_same_index = check_zeros_same_index(image)

        # check distance is 0 in main branch
        main_branch_zero = check_zero_dist_main_branch(image)

        # check no gaps between columns
        inconsistent_branch, where_branch, where_chan= check_gaps_between_branches(image)

        # check no gaps between subhalo in same branch
        inconsistent_gap, where_gap_bran, where_gap_row = check_gaps_in_branch(image)
        
        last_descendant_exist = check_last_descendant(image)

        # check if image is overall consistant and add to consistant images
        if zero_same_index == 1 and main_branch_zero == 1 and not inconsistent_branch and not inconsistent_gap and last_descendant_exist:
            consistant_images.append(image)

        else:
            inc = 0
            inconsistent_images.append(image)
            if print_index:
                if zero_same_index != 1:
                    print(f" image index = {i}, inconsistent because zero not same index: zero in {(100 * zero_same_index):.2f}% of same spots across channels")
                    zero_inconsistency += 1
                    inc += 1
                if main_branch_zero != 1:
                    print(f" image index = {i}, inconsistent because dist in main branch is not zero: zero in {(100 * main_branch_zero):.2f}% of distance main branch")
                    dist_main_branch += 1
                    inc += 1
                if inconsistent_branch:
                    print(f" image index = {i}, inconsistent because gaps between branches {where_branch - 1} and {where_branch + 1} in channel {where_chan}")
                    gap_between_branch += 1
                    inc += 1
                if inconsistent_gap:
                    print(f" image index = {i}, inconsistent because gaps in branch {where_gap_bran} at row {where_gap_row + 1}")
                    gap_in_branch += 1
                    inc += 1
                if not last_descendant_exist:
                    print(f" image index = {i}, inconsistent it does not have a last descendant")
                    last_descendant += 1
                    inc += 1
            
            else:
                if zero_same_index != 1:
                    zero_inconsistency += 1
                    inc += 1
                if main_branch_zero != 1:
                    dist_main_branch += 1
                    inc += 1
                if inconsistent_branch:
                    gap_between_branch += 1
                    inc += 1
                if inconsistent_gap:
                    gap_in_branch += 1
                    inc += 1
                if not last_descendant_exist:
                    last_descendant += 1
                    inc += 1
            
            if inc > 1:
                two_or_more_inconsistencies +=1
    
    if len(consistant_images) != 0:
        consistant_images = torch.stack(consistant_images).squeeze(1)
        
        perc = 100 * consistant_images.shape[0] / num_images
    
        print("\n")
        print(f"Percentage of consistant images = {perc:.2f}%")
        
    if len(consistant_images) == 0:
        print("\n")
        print(f"Percentage of consistant images = {0.0}%")
    
    if len(inconsistent_images) != 0:
        inconsistent_images = torch.stack(inconsistent_images).squeeze(1)
        
        inconsistant = len(inconsistent_images)
        print("\nInconsistency reasons:")
        print(f"inconsistency due to zero / nonzero mistake = {(zero_inconsistency * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to distance not zero in main branch =  {(dist_main_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to gap between branches =  {(gap_between_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to zgap in branch {(gap_in_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to last descendant dont exist {(last_descendant * 100 / inconsistant):.2f}%")

        print(f"\nNumber of images with two or more inconsistencies = {two_or_more_inconsistencies}, which is  {(two_or_more_inconsistencies * 100 / inconsistant):.2f}%")
        print(f"That corresponds to {(two_or_more_inconsistencies * 100 / num_images):.2f}% of all images")
        
        print("\n")
        print(f"Of all images, {(zero_inconsistency * 100 / num_images):.2f}% have zero inconsistency")
        print(f"Of all images, {(dist_main_branch * 100 / num_images):.2f}% have distance main branch inconsistency")
        print(f"Of all images, {(gap_between_branch * 100 / num_images):.2f}% have gap between branches inconsistency")
        print(f"Of all images, {(gap_in_branch * 100 / num_images):.2f}% have gap within branch inconsistency")
        print(f"Of all images, {(last_descendant * 100 / num_images):.2f}% have last descendant inconsistency")
    
    return consistant_images, inconsistent_images


def check_branch_length(dataset, printer = True):
    one_branch = []
    two_branch = []
    three_branch = []
    four_branch = []
    five_branch = []
    six_branch = []
    seven_branch = []
    eight_branch = []
    nine_branch = []
    ten_branch = []

    for datapoint in dataset:
        one_channel = datapoint[1]
        branches = torch.count_nonzero(one_channel, dim=0)
        num_branch = torch.count_nonzero(branches, dim=0)
        if num_branch == 1:
            one_branch.append(datapoint)
        elif num_branch == 2:
            two_branch.append(datapoint)
        elif num_branch == 3:
            three_branch.append(datapoint)
        elif num_branch == 4:
            four_branch.append(datapoint)
        elif num_branch == 5:
            five_branch.append(datapoint)
        elif num_branch == 6:
            six_branch.append(datapoint)
        elif num_branch == 7:
            seven_branch.append(datapoint)
        elif num_branch == 8:
            eight_branch.append(datapoint)
        elif num_branch == 9:
            nine_branch.append(datapoint)
        elif num_branch == 10:
            ten_branch.append(datapoint)

    total = [one_branch, two_branch, three_branch, four_branch, five_branch,
             six_branch, seven_branch, eight_branch, nine_branch, ten_branch]

    total_im = 0
    total_branches = 0
    for i, branch_list in enumerate(total):
        total_im += len(branch_list)
        total_branches += (i + 1) * len(branch_list)

    avg_branch = total_branches / total_im
    if printer:
        print(f"\naverage number of branches in the image = {avg_branch:.2f} vs. 7.12 in training data")
    
    nonzero_indices = torch.nonzero(dataset[:, 1].flatten())
    nonzero_value = len(dataset[:, 1].flatten()[nonzero_indices[:, 0]])

    average_nonzero_entries = nonzero_value / total_im 
    avg_branch_length = nonzero_value / total_branches
    if printer:
        print(f"Average branch length = {avg_branch_length:.2f} vs. 9.06 in training data")
        print(f"Number of nonzero entries (progenitors) = {average_nonzero_entries:.2f} vs. 64.55 in training data")
        print("\n\n", flush=True)

    return total

def draw_sample_given_number_branch(dataset, num_branches):

    total = check_branch_length(dataset, printer = False)
    merger_trees_given_branch_length = total[num_branches - 1]
    i = random.randint(0, len(merger_trees_given_branch_length))
    print(f"Sampling a generated merger tree with {num_branches} branches")
    print(f"\nPicked random sample number {i} out of {len(merger_trees_given_branch_length)} potential samples")
    sample = merger_trees_given_branch_length[i].unsqueeze(0)

    return sample

def draw_sample_given_complexity(dataset, higher_than, threshold):

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
                check_branch_length(sample.unsqueeze(0))
                return sample
            else:
                continue
        
    print("Could not find a tree with the desired complexity")

    print("Generating random sample:")
    sample = dataset[i]
    return sample
    

def draw_sample_given_branch_and_complexity(dataset, num_branches = None, threshold = None, higher_than = False):

    if num_branches is None and threshold is None:
        print("You must specify either a given number of branches or a threshold, generating random merger tree without any specifications:")
        i = random.randint(0, len(dataset))
        sample = dataset[i]
        return sample.unsqueeze(0)

    elif num_branches is not None:
        print(f"Sampling a generated merger tree with {num_branches} branches")

        total = check_branch_length(dataset, printer = False)
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
    plt.hist(distribution.numpy(), bins=100, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('distributions of all Values')
    plt.show()
    print(f"number of nonzero values = {len(nonzero_value)}")
    
    return distribution, nonzero_distribution


    
def get_large_sample_ks_significance_level(dist1, dist2, alpha):
    m = len(dist1.flatten())
    n = len(dist2.flatten())
    
    sig_level = np.sqrt(-np.log(alpha / 2) * ((1 + (m / n)) / (2 * m)))
    
    return sig_level


    
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
    print(f"\nAccording to regular significance level of {alpha}:")
    if p_value > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
        
     # Interpret the results
    sig_level = get_large_sample_ks_significance_level(real_dist, fake_dist, alpha)  # significance level
    print(f"\nAccording to large sample significance level of {alpha}, giving significance level of {sig_level:.4f}:")
    if statistic < sig_level:
        print(f"The distributions are not significantly different (fail to reject H0), KS statistic {statistic:.4f} < {sig_level:.4f}")
    else:
        print(f"The distributions are significantly different (reject H0), KS statistic {statistic:.4f} > {sig_level:.4f}")
       
    
        
    
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
    print(f"\nAccording to regular significance level of {alpha}:")
    if p_value_nonzero > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
        
    # Interpret the results
    sig_level = get_large_sample_ks_significance_level(real_nonzero_dist, fake_nonzero_dist, alpha)  # significance level
    print(f"\nAccording to large sample significance level of {alpha}, giving significance level of {sig_level:.4f}:")
    if statistic_nonzero < sig_level:
        print(f"The distributions are not significantly different (fail to reject H0), KS statistic {statistic_nonzero:.4f} < {sig_level:.4f}")
    else:
        print(f"The distributions are significantly different (reject H0), KS statistic {statistic_nonzero:.4f} > {sig_level:.4f}")
       
    

    return statistic, statistic_nonzero

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

"""
def analyze_data(generated_data_path, comparing_data_path):
    generated_data = torch.load(generated_data_path, map_location = "cpu")
    original_data = torch.load(comparing_data_path)
    if original_data.shape[0] == 3:
        original_data = original_data.permute(1, 0, 2, 3)
    
    print("\nPlot images:")
    images = plot_images(generated_data, 10)
    
    print("\nCheck consistency:")
    consistant, inconsistant = check_consistency(generated_data)
    
    print("\nCheck number of branches of all generated images:")
    check_branch_length(generated_data)
    
    if len(consistant) != 0:
        print("\nCheck number of branches of consistent generated images:")
        check_branch_length(consistant)
    
    print("\nMass KS test against training data:")
    ks, ks_nonzero = ks_test(original_data, generated_data, dim = 1)
    
    return generated_data, consistant, inconsistant
"""

def minmax_norm(img):
    minmax_data = []
    chan = img.shape[1]
    for im in img:
        new_data = im.clone()
        for i in range(chan):
            min_value = im[i].min()
            max_value = im[i].max()

            normalized_data = (im[i] - min_value) / (max_value - min_value)

            new_data[i] = normalized_data
        
        minmax_data.append(new_data.unsqueeze(0))
    
    processed_data = torch.cat(minmax_data).detach()
        
    return processed_data

def diffusion_train_epoch(model, optimizer, batch, lr, T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    bs = batch.shape[0]

    batch = batch.to(dtype=torch.float32)

    optimizer.zero_grad()

    t = torch.randint(0, T, (bs,), device=device).long()
    loss, batch_noisy = get_loss(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    loss.backward()
    optimizer.step()

    return batch_noisy, loss, model

def training_latent_diffusion(encoder, generator, latent_diffusion_model, dataloader,
                              num_epochs, lr, T, device, pretrained = True):
    
    encoder, generator, latent_diffusion_model = encoder.to(device), generator.to(device), latent_diffusion_model.to(device)
    optimizer = Adam(latent_diffusion_model.parameters(), lr = lr)
    
    transform = transforms.Resize((64, 64))
    transform_back = transforms.Resize((10, 10))
    
    betas = linear_beta_schedule(timesteps=T)
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1} / {num_epochs}")
        
        
        for batch in tqdm.tqdm(dataloader):
            # (bs, 3, 29, 10)
            #print("should be (bs, 3, 29, 10):", batch.shape)
            batch = batch.to(dtype=torch.float32).to(device)
            
            # run through encoder
            # (bs, 300)
            latent_representation = encoder(batch)
            #print("should be (bs, 300):",latent_representation.shape)
            # reshape
            # (bs, 3, 10, 10)
            latent_review = latent_representation.view(-1, 3, 10, 10)
            #print("should be (bs, 3, 10, 10):", latent_review.shape)
            # (bs, 3, 64, 64)
            latent_reshape = transform(latent_review)
            #print("should be (bs, 3, 64, 64):", latent_reshape.shape)
            
            #run through diffusion
            # (bs, 3, 64, 64)
            latent_diffused, loss, latent_diffusion_model = diffusion_train_epoch(latent_diffusion_model,
                                                                                  optimizer,
                                                                                  latent_reshape,
                                                                                  lr,
                                                                                  T,
                                                                                  sqrt_alphas_cumprod,
                                                                                  sqrt_one_minus_alphas_cumprod,
                                                                                  device)
            #print("should be (bs, 3, 64, 64):", latent_diffused.shape)
            
            #reshape
            # (bs, 3, 10, 10)
            latent_out = transform_back(latent_diffused)
            #print("should be (bs, 3, 10, 10):", latent_out.shape)
            # (bs, 300)
            latent_out_view = latent_out.view(-1, 300)
            #print("should be (bs, 300):", latent_out_view.shape)
            #run through generator
            output = generator(latent_out_view)
            #print("should be (bs, 3, 29, 10):", output.shape)
            
        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
        
        
        plt.imshow(batch[0].detach().permute(1, 2, 0))
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        plt.imshow(latent_reshape[0].detach().permute(1, 2, 0))
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        plt.imshow(latent_diffused[0].detach().permute(1, 2, 0))
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()

        plt.imshow(output[0].detach().permute(1, 2, 0))
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()
        
    return encoder, generator, latent_diffusion_model
            

@torch.no_grad()
def sample_timestep2(model, x, t, betas):
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
        img = sample_timestep2(model, img, t, betas)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    return img


class LatentDiffusionModel_mlp(nn.Module):
    def __init__(self, input_size = 300, hidden_size = 128, output_size = 300):
        super(LatentDiffusionModel_mlp, self).__init__()
        self.time_emb_dim = hidden_size

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Define MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 3)
        self.fc3 = nn.Linear(hidden_size * 3, hidden_size * 4)
        self.fc4 = nn.Linear(hidden_size * 4, hidden_size * 3)
        self.fc5 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc7 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x, t):
        time_emb = self.time_mlp(t)
        x0 = self.relu(self.fc1(x))
        x1 = torch.cat([x0, time_emb], dim = 1)
        x2 = self.relu(self.fc2(x1))
        x3 = self.relu(self.fc3(x2))
        x4 = self.relu(self.fc4(x3)) + x2
        x5 = self.relu(self.fc5(x4)) + x1
        x6 = self.relu(self.fc6(x5)) + x0
        x7 = self.fc7(x6)
        return x
    
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
    plt.colorbar(sm, label="mass value")
    
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