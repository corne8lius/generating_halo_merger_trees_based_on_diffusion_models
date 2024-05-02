from torch.utils.data import DataLoader
import torch
import warnings
import time
warnings.filterwarnings('ignore')

from GAN_model import Encoder, Generator
from diffusion_model import SimpleUnet
from latent_diffusion_functions import create_generated_images_latent_diffusion, training_latent_diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Latent diffusion 1.0")
fox = True
training = True
train_encoder_decoder = False
generate = False

batch_size = 256
epochs = 200
T = 1000
img_size = 64
lr = 3e-4
print("latent diffusion 1.0", flush=True)
print("Loading data ...")
if fox:
    data_path = "/fp/projects01/ec35/homes/ec-corneb/data/dataset_distlognorm_massnorm.pt"
    generator_path = "/fp/projects01/ec35/homes/ec-corneb/models/fox_attempt1/generator_fox_test.pt"
    encoder_path = "/fp/projects01/ec35/homes/ec-corneb/models/fox_attempt1/encoder_fox_test.pt"
else:
    epochs = 1
    T = 2
    data_path = "../notebooks/data/dataset_distlognorm_massnorm.pt"
    generator_path = "../notebooks/models/fox_attempt1/generator_fox_test.pt"
    encoder_path = "../notebooks/models/fox_attempt1/encoder_fox_test.pt"

dataset = torch.load(data_path)
if fox:
    train_loader = DataLoader(dataset, shuffle=True, batch_size = batch_size)
else:
    train_loader = DataLoader(dataset[:batch_size], shuffle=True, batch_size = batch_size)
print("Loading data done")

print("Loading models ...")
saved_generator = torch.load(generator_path, map_location = device)
generator_state_dict = saved_generator["generator"]
latent_dim = saved_generator["latent_space"]
num_branches = saved_generator["num_branches"]
nvar = saved_generator["nvar"]

generator = Generator(nvar, num_branches, latent_dim)
generator.load_state_dict(generator_state_dict)

saved_encoder = torch.load(encoder_path, map_location = device)
encoder_state_dict = saved_encoder["encoder"]

encoder = Encoder(nvar, num_branches, latent_dim)
encoder.load_state_dict(encoder_state_dict)

latent_diffusion = SimpleUnet()
print("Loading models done")


if training:
    print("Training model ...")
    for param in generator.parameters():
        param.requires_grad = train_encoder_decoder
        
    for param in encoder.parameters():
        param.requires_grad = train_encoder_decoder

    start_time = time.time()
    encoder_test, decoder_test, latent_diffusion_model_test = training_latent_diffusion(encoder,
                                                                                        generator,
                                                                                        latent_diffusion,
                                                                                        train_loader,
                                                                                        epochs,
                                                                                        lr,
                                                                                        T,
                                                                                        device,
                                                                                        fox,
                                                                                        pretrained = True)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Training time:", elapsed_time, "seconds", flush=True)
    print("Training model done")


if generate:
    print("Generating sampled dataset ...")
    if fox:
        latent_diffusion_name= "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/latent_diffusion_model1.0.pt"
        generator_name = "/fp/projects01/ec35/homes/ec-corneb/diffusion/latent_diffusion/generator_model1.0.pt"
    else: 
        latent_diffusion_name= "../notebooks/diffusion_notebook/latent_diffusion/1.0/latent_diffusion_model1.0.pt"
        generator_name = "../notebooks/diffusion_notebook/latent_diffusion/1.0/generator_model1.0.pt"
    

    latent_diffusion_model = SimpleUnet()
    generator = Generator(nvar, num_branches, latent_dim)

    saved_generator_model = torch.load(generator_name, map_location = device)
    generator.load_state_dict(saved_generator_model)

    saved_latent_model = torch.load(latent_diffusion_name, map_location = device)
    latent_diffusion_model.load_state_dict(saved_latent_model)

    reference_shape = dataset[:10000].shape

    start_time = time.time()
    generated_latent_images = create_generated_images_latent_diffusion(latent_diffusion_model, generator, img_size,
                                                                       reference_shape, batch_size, device, T = 1000)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Generating 10k images time:", elapsed_time, "seconds", flush=True)

    if fox:
        print("saving data ...")
        save_data_path = "../diffusion/latent_diffusion/1.0/latent_diffusion_images1.2.pt"
        torch.save(generated_latent_images, save_data_path)
        print(f"\tlssaved generated data as", save_data_path)
        print("saving data done")
    print("Generating sampled dataset done")

