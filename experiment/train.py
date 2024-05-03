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

from useful_functions import save_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################################################################
#                           GAN
#######################################################################################

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


def train_GAN(train_dataset, discriminator_model, encoder_model, generator_model,
              num_train_gen, noise_uniform, recon_scale, args,  num_epochs = 200,
              batch_size = 128, latent_dim = 300, lr = 3e-4):
    nsnap = train_dataset.shape[2]
    nbr = train_dataset.shape[3]
    nvar = train_dataset.shape[1]

    dataloader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)

    discriminator = discriminator_model(nvar, nbr).to(device)
    encoder = encoder_model(nvar, nbr, latent_dim).to(device)
    generator = generator_model(nvar, nbr, latent_dim).to(device)

    optimizer_enc_dec = optim.AdamW(list(encoder.parameters()) + list(generator.parameters()), lr=lr)
    optimizer_discriminator = optim.AdamW(discriminator.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    reconstructed_images_training = []
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1} / {num_epochs}")
        if epoch % num_train_gen == 0: 
            print("Training generator")
            
        discriminator.train()
        encoder.train()
        generator.train()
        for real_images in tqdm.tqdm(dataloader):
            bs = real_images.shape[0]
            # Train Discriminator
            discriminator.zero_grad()
            real_images = real_images.to(dtype=torch.float32)
            real_images = real_images.to(device)

            # Discriminator forward pass with real images
            real_output = discriminator(real_images).view(-1, 1)
            
            # Generate fake images
            if noise_uniform:
                high = 1.0
                low = -1.0
                uniform_noise = torch.rand(bs, latent_dim) * (high - (low)) + (low)
                noise = uniform_noise.to(device)
            
            else:
                noise = torch.randn(bs, latent_dim).to(device)
            fake_images = generator(noise).to(device)

            inputs = torch.cat([real_images, fake_images.detach()]).to(device)
            labels = torch.cat([torch.ones(bs, 1), torch.zeros(bs, 1)]).to(device) # fake
            
            # Discriminator forward pass with fake images

            fake_output = discriminator(fake_images.detach()).view(-1, 1)
            output = discriminator(inputs).to(device)

            fake_loss = criterion(output, labels)
            
            # Backpropagate and update discriminator weights
            d_loss = fake_loss
            d_loss.backward()
            optimizer_discriminator.step()
            
            if epoch % num_train_gen == 0 or epoch + 1 == num_epochs: 
                # Train Generator (Encoder + Decoder)
                encoder.zero_grad()
                generator.zero_grad()

                recon = encoder(real_images)
                recon_images = generator(recon)
                output = discriminator(fake_images).view(-1, 1).to(device)

                # Generator loss
                real_labels = torch.ones(bs, 1).to(device)
                g_loss = criterion(output, real_labels)
                    
                loss_rec = custom_reconstruction_loss(real_images, recon_images) * recon_scale

                loss_g = g_loss + loss_rec
                # Backpropagate and update generator weights
                loss_g.backward()
                optimizer_enc_dec.step()

            if epoch + 1 == num_epochs:
                reconstructed_images_training.append(recon_images)

        if epoch % 50 == 0 or epoch + 1 == num_epochs:
            if args.fox:
                disc_name = f"../models/10_branches/discriminator_retrain_epoch{epoch}.pt"
                gen_name = f"../models/10_branches/generator_retrain_epoch{epoch}.pt"
                enc_name = f"../models/10_branches/encoder_retrain_epoch{epoch}.pt"
            else:
                disc_name = "../notebooks/models/attempt3/discriminator_local_test.pt"
                gen_name = "../notebooks/models/attempt3/generator_local_test.pt"
                enc_name = "../notebooks/models/attempt3/encoder_local_test.pt"  
            save_model(encoder, generator, discriminator,
                    optimizer_discriminator, optimizer_enc_dec,
                    disc_name, gen_name, enc_name,
                    nbr, nsnap, nvar, latent_dim)
            
            print(f"saved at epoch: {epoch + 1} / {num_epochs}")
            
            # Print losses
        if epoch % 10 == 0 or epoch + 1 == num_epochs:
            with torch.no_grad():
                if not args.fox:
                    discriminator.eval()
                    encoder.eval()
                    generator.eval()

                    print("real image:")
                    test = real_images[0]
                    img = test.permute(1, 2, 0).detach().cpu().numpy()
                    plt.ioff()
                    plt.imshow(img)
                    plt.axis('off')  # Optional: Turn off axis ticks and labels
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close("all")

                    print("reconstructed image:")
                    real = real_images.to(dtype=torch.float32)
                    encoded = encoder(real)
                    reconstructed = generator(encoded)
                    recon = reconstructed.reshape(-1, nvar, nsnap, nbr)
                    img = recon[0].permute(1, 2, 0).detach().cpu().numpy()
                    plt.ioff()
                    plt.imshow(img)
                    plt.axis('off')  # Optional: Turn off axis ticks and labels
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close("all")


                    # generate and show fake images
                    print("random created images:")
                    fixed_noise = torch.randn((batch_size, latent_dim)).to(device)
                    fake = generator(fixed_noise).reshape(-1, nvar, nsnap, nbr)
                    im = random.randint(0, batch_size - 3)
                    img = fake[im : im + 2]
                    #print(img[0])
                    for im in img:
                        img = im.permute(1, 2, 0).detach().cpu().numpy()
                        plt.ioff()
                        plt.imshow(img)
                        plt.axis('off')  # Optional: Turn off axis ticks and labels
                        plt.show(block=False)
                        plt.pause(1)
                        plt.close("all")
                    
                print(f"Epoch [{epoch}/{num_epochs}],"
                    f"Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}")

                print(real_output[:3])
                print(fake_output[:3])
                print("mean disc rating of real images:", torch.mean(real_output))
                print("mean disc rating of fake images:", torch.mean(fake_output))
                
    reconstructed_images_training = torch.cat(reconstructed_images_training)
    print("reconstructe images stored as \"reconstructed_images_training\"")

    return reconstructed_images_training
        


#######################################################################################
#                           DIFFUSION
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

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Resize((29, 10)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0))
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, AE = True):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    
    if AE:
        encoder = model[0].to(device)
        decoder = model[1].to(device)
        latent = encoder(x_noisy, t)
        noise_pred = decoder(latent, t)
    else:
        model.to(device)
        noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def diffusion_train(model, dataloader, epochs, batch_size, lr, T, device, args, save_name = "../diffusion/autoencoder/diffusion_test1.pt"):
    
    if args.AE:
        encoder = model[0].to(device)
        decoder = model[1].to(device)
        optimizer =  optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    else:
        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)

    betas = linear_beta_schedule(timesteps=T)
    # Pre-calculate different terms for closed form
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    for epoch in range(epochs):
        step = 0
        print(f"epoch {epoch + 1} / {epochs}")
        for batch in tqdm.tqdm(dataloader):
            bs = batch.shape[0]
            
            batch = batch.to(dtype=torch.float32)
            
            optimizer.zero_grad()

            t = torch.randint(0, T, (bs,), device=device).long()
            loss = get_loss(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, args.AE)
            loss.backward()
            optimizer.step()
            step += 1

        if epoch % 50 == 0 or epoch == epochs - 1 or epoch == 0:
            if args.AE and args.fox:
                encoder_name = "../diffusion/autoencoder/encoder.pt"
                decoder_name = "../diffusion/autoencoder/decoder.pt"
                torch.save(decoder.state_dict(), decoder_name)
                torch.save(encoder.state_dict(), encoder_name)
                print(f"saved AE diffusion model")
            elif args.fox:
                torch.save(model.state_dict(), save_name)
                print(f"saved diffusion model as {save_name}")
            elif args.save:
                save_name = "../notebooks/diffusion_notebook/autoencoder/diffusion_test_AE.pt"
                torch.save(model.state_dict(), save_name)
                print(f"saved diffusion model as {save_name}")

        print(f"Epoch {epoch + 1} | step {step:03d} Loss: {loss.item()} ")

    if args.AE:
        model = [encoder, decoder]
    return model
