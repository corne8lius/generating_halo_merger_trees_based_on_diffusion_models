import torch.nn.functional as F
from torchvision import transforms 
import torch
from torch import nn
from torch.optim import Adam, AdamW
import tqdm
import math

from diffusion_model import Diffusion
from useful_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
    var = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
    return mean + var, noise.to(device)

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    
    model.to(device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred), x_noisy

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
                              num_epochs, lr, T, device, fox, pretrained = True):
    
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

        if epoch % 10 == 0:
            if fox:
                latent_diffusion_name= "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/1.0/latent_diffusion_model1.0.pt"
                encoder_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/1.0/encoder_model1.0.pt"
                generator_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/1.0/generator_model1.0.pt"
                torch.save(latent_diffusion_model.state_dict(), latent_diffusion_name)
                torch.save(encoder.state_dict(), encoder_name)
                torch.save(generator.state_dict(), generator_name)
                print("models saved")

        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")  
        
    return encoder, generator, latent_diffusion_model


def diffusion_train_epoch2(images, diffusion, model, optimizer, mse, device):
    t = diffusion.sample_timesteps(images.shape[0]).to(device)
    x_t, noise = diffusion.noise_images(images, t)
    x_t, noise = x_t.to(device), noise.to(device)
    predicted_noise = model(x_t, t).to(device)
    loss = mse(noise, predicted_noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return predicted_noise, loss, diffusion, model

def training_latent_diffusion2(encoder, generator, latent_diffusion_model, dataloader,
                               img_size, num_epochs, lr, T, device, fox, pretrained = True):
    
    encoder, generator, latent_diffusion_model = encoder.to(device), generator.to(device), latent_diffusion_model.to(device)
    optimizer = Adam(latent_diffusion_model.parameters(), lr = lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(T = T, img_size=img_size, device=device)
    l = len(dataloader)
    
    transform = transforms.Resize((64, 64))
    transform_back = transforms.Resize((10, 10))

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
            latent_diffused, loss, diffusion, latent_diffusion_model = diffusion_train_epoch2(latent_reshape,
                                                                                             diffusion,
                                                                                             latent_diffusion_model,
                                                                                             optimizer,
                                                                                             mse,
                                                                                             device)
            #print("should be (bs, 3, 64, 64):", latent_diffused.shape)
            
            #reshape
            # (bs, 3, 10, 10)
            latent_out = transform_back(latent_diffused)
            #print("should be (bs, 3, 10, 10):", latent_out.shape)
            # (bs, 300)
            latent_out_view = latent_out.reshape(-1, 300)
            #print("should be (bs, 300):", latent_out_view.shape)
            #run through generator
            output = generator(latent_out_view)
            #print("should be (bs, 3, 29, 10):", output.shape)

        if epoch % 10 == 0:
            if fox:
                latent_diffusion_name= "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/2.0/latent_diffusion_model2.0.pt"
                encoder_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/2.0/encoder_model2.0.pt"
                generator_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/2.0/generator_model2.0.pt"
                torch.save(latent_diffusion_model.state_dict(), latent_diffusion_name)
                torch.save(encoder.state_dict(), encoder_name)
                torch.save(generator.state_dict(), generator_name)
                print("models saved")
            
        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
        
    return encoder, generator, latent_diffusion_model

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
def sample_image(model, noise, T):
    img = noise
    betas = linear_beta_schedule(timesteps=T)
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, betas)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    return img
            

def create_generated_images_latent_diffusion(model, generator, img_size, reference_shape, batch_size, device, T = 1000):
    model = model.to(device)
    generator = generator.to(device)
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 128
    runs = math.floor(num_images / batch_size)
    extra = num_images % batch_size
    transform = transforms.Resize((64, 64))
    transform_back = transforms.Resize((10, 10))
    
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

        noise = torch.randn((batch_size, 300), device=device)
        noise_view = noise.reshape(-1, 3, 10, 10)
        noise_reshape = transform(noise_view)
        out_test = sample_image(model, noise_reshape, T)
        latent_out = transform_back(out_test)
        latent_out_view = latent_out.reshape(-1, 300).to(device)
        output = generator(latent_out_view)
            
        generated_images.append(output)
    
    images = torch.cat(generated_images).detach()
    images = transform(images)
    
    return images


    
def diffusion_train_epoch_latent(model, optimizer, batch, lr, T, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device):
    bs = batch.shape[0]

    batch = batch.to(dtype=torch.float32)

    optimizer.zero_grad()

    t = torch.randint(0, T, (bs,), device=device).long()
    loss, batch_noisy = get_loss(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

    return batch_noisy, loss, model

def training_latent_diffusion_latent(encoder, generator, latent_diffusion_model, dataloader,
                              num_epochs, lr, T, device, pretrained = True):
    torch.set_grad_enabled(True)
    
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
            
            #run through diffusion
            # (bs, 3, 64, 64)
            latent_diffused, loss, latent_diffusion_model = diffusion_train_epoch_latent(latent_diffusion_model,
                                                                                          optimizer,
                                                                                          latent_representation,
                                                                                          lr,
                                                                                          T,
                                                                                          sqrt_alphas_cumprod,
                                                                                          sqrt_one_minus_alphas_cumprod,
                                                                                          device)
            
            #run through generator
            output = generator(latent_diffused)
            #print("should be (bs, 3, 29, 10):", output.shape)

        if epoch % 10 == 0:
            if True:
                latent_diffusion_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/mlp/latent_diffusion_model_mlp.pt"
                encoder_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/mlp/encoder_model_mlp.pt"
                generator_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/mlp/generator_model_mlp.pt"
                torch.save(latent_diffusion_model.state_dict(), latent_diffusion_name)
                torch.save(encoder.state_dict(), encoder_name)
                torch.save(generator.state_dict(), generator_name)
                print("models saved")
            
        print(f"Epoch {epoch + 1} | Loss: {loss.item()} ")
        
    return encoder, generator, latent_diffusion_model



@torch.no_grad()
def sample_timestep_mlp(model, x, t, betas):
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

    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_image_mlp(model, noise, T, bs):
    img = noise
    betas = linear_beta_schedule(timesteps=T)
    for i in range(0, T)[::-1]:
        t = torch.full((bs,), i, device=device, dtype=torch.long)
        img = sample_timestep_mlp(model, img, t, betas)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

    return img


def create_generated_images_latent_mlp_diffusion(model, generator, reference_shape, batch_size, device, T = 1000):
    model = model.to(device)
    generator = generator.to(device)
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 128
    runs = math.floor(num_images / batch_size)
    extra = num_images % batch_size

    for run in range(runs + 1):
        if run % 1 == 0:
            print(f"epoch {run} / {runs}")
        if run == runs:
            batch_size = extra

        noise = torch.randn((batch_size, 300), device=device)
        out_test = sample_image_mlp(model, noise, T, batch_size)
        output = generator(out_test)
            
        generated_images.append(output)
    
    images = torch.cat(generated_images).detach()
    
    return images