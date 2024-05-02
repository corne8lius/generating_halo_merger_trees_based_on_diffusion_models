import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
from torch import nn
import tqdm
import logging
from torch import optim
from torchvision import transforms
import math
import time
import warnings
import os
import random
import numpy as np
warnings.filterwarnings('ignore')

from useful_functions import create_generated_images_diffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed_value=1238):
    "Set same seed to all random operations for reproduceability purposes"
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

seed_everything(1238)

fox = True
training = True
generate = True

batch_size = 16
epochs = 400
T = 950
img_size = 64
lr = 3e-4
linear = True
#interpolation_mode = ("NEAREST", transforms.InterpolationMode.NEAREST)
#interpolation_mode = ("NEAREST_EXACT", transforms.InterpolationMode.NEAREST_EXACT)
interpolation_mode = ("BILINEAR", transforms.InterpolationMode.BILINEAR)
#interpolation_mode = ("BICUBIC", transforms.InterpolationMode.BICUBIC)
interpolation = interpolation_mode[1]
interpolation_name = interpolation_mode[0]
T_name = str(T)


if linear:
    print("scheudler = Linear")
    scheduler = "linear"
else:
    print("scheudler = Cosine")
    scheduler = "cosine"
print("Interpolation mode = ", interpolation_name)
print("T = ", T)
print("Epochs =", epochs)

name = "epochs=" + str(epochs) + "_" + interpolation_name + "_" + "T=" + T_name + "_" + scheduler + ".pt"

if fox:
    #data_path = "/fp/projects01/ec35/homes/ec-corneb/data/dataset_distlognorm_massnorm.pt"
    data_path = "/fp/projects01/ec35/homes/ec-corneb/data/dataset_normalized_consistent_only.pt"
else:
    epochs = 1
    T = 2
    data_path = "../notebooks/data/dataset_distlognorm_massnorm.pt"

dataset = torch.load(data_path)
transform = transforms.Resize((img_size, img_size), interpolation = interpolation)
resized_image = transform(dataset)
if fox:
    train_loader = DataLoader(resized_image, shuffle=True, batch_size = batch_size)
else:
    train_loader = DataLoader(resized_image[:16], shuffle=True, batch_size = batch_size)

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
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        if linear:
            print("Using linear scheduler")
            self.beta = self.prepare_noise_schedule().to(device)
        else:
            print("Using cosine scheduler")
            self.beta = self.cosine_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.T)
    
    def cosine_schedule(self, s=0.008):
        def f(t, s):
            return torch.cos((t / self.T + s) / (1 + s) * 0.5 * torch.pi) ** 2
        x = torch.linspace(0, self.T, self.T + 1)
        alphas_cumprod = f(x, s) / f(torch.tensor([0]), s)
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(betas, 0.0001, 0.999)
        return betas

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.T, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm.tqdm(reversed(range(1, self.T)), position=0):
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
        self.time_dim = torch.tensor(time_dim).to(device)
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
        ).to(device)
        pos_enc_a = torch.sin(t.repeat(1, channels.to(device) // 2) * inv_freq).to(device)
        pos_enc_b = torch.cos(t.repeat(1, channels.to(device) // 2) * inv_freq).to(device)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
            t = t.unsqueeze(-1).type(torch.float).to(device)
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
    
def custom_reconstruction_loss(output, target, scale = 10):
    # Calculate the reconstruction loss
    recon_loss = F.mse_loss(output, target, reduction='none')
    
    # Apply higher penalty if target is zero
    penalty = torch.where((target == 0.0) |(target == 0.5) | (target == 1.0), scale * recon_loss, recon_loss)
    
    # Calculate the mean loss
    mean_loss = torch.mean(penalty)
    
    return mean_loss


def train(dataloader, epochs, bs, img_size, lr, device):
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(T = T, img_size=img_size, device=device)
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
            loss = mse(noise, predicted_noise)
            #loss = custom_reconstruction_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()

        loss_list.append(loss_epoch / len(dataloader))
        sampled_images = diffusion.sample(model, n=1)
        
        if epoch % 5 == 0:
            if fox:
                model_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other/" + name
            else:
                model_name = "../notebooks/diffusion_notebook/T&E1000/epoch1000/diffusion_test.pt"
            torch.save(model.state_dict(), model_name)
            print("Saved model as: ", model_name)

        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
        
    return loss_list, sampled_images, model

if training:     
    print("Training ...")
    start_time = time.time()
    loss_list, sampled_images, model = train(train_loader, epochs, batch_size, img_size, lr, device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", elapsed_time, "seconds", flush=True)
    print("Training done ...")

def create_generated_images_diffusion2(model, img_size, reference_shape, T):
    model = model.to(device)
    diffusion = Diffusion(T, img_size=img_size, device=device)
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


if generate:
    print("Generating data ...")
    T = T
    model = UNet()

    if fox:
        model_path = "/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other/" + name
    else:
        model_path = "../notebooks/diffusion_notebook/T&E1000/epoch1000/diffusion_test.pt"
    
    saved_model = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(saved_model)
    if fox:
        reference_shape = dataset[:3000].shape
    else:
        reference_shape = dataset[0:1].shape
    start_time = time.time()
    generated = create_generated_images_diffusion2(model, img_size, reference_shape, T = T)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Generating 10k images time:", elapsed_time, "seconds", flush=True)

    if fox:
        save_data_path = "/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other/generated_image_" + name
    else:
        save_data_path = "../notebooks/diffusion_notebook/diffusion_model2.0_images.pt"
    torch.save(generated, save_data_path)
    print(f"saved generated data as", save_data_path)

    print("Generating data done")