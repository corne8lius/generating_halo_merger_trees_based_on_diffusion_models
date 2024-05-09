import torch
import torch.nn as nn
import tqdm
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms 
from torch.optim import Adam

from experiment.diffusion_model import UNet, Diffusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataloader, model, diffusion_model, epochs, T, img_size, lr, name, device):
    """
    training function for the diffusion model
    """
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()

        loss_list.append(loss_epoch / len(dataloader))
        sampled_images = diffusion.sample(model, n=1)
        
        if epoch % 5 == 0:
            model_name = "diffusion/diffusion_test/model/" + name
            torch.save(model.state_dict(), model_name)
            print("Saved model as: ", model_name)

        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4} ")
        
    return loss_list, sampled_images, model