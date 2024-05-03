from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')

from GAN_model import Encoder, Generator
from diffusion_model import UNet
from latent_diffusion_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fox = True
training = True
train_encoder_decoder = False
generate = True

batch_size = 16
epochs = 200
T = 1000
img_size = 64
lr = 3e-4

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

model = UNet()
print("Loading models done")


if training:
    print("Training model ...")
    for param in generator.parameters():
        param.requires_grad = train_encoder_decoder
        
    for param in encoder.parameters():
        param.requires_grad = train_encoder_decoder

    encoder, generator, latent_diffusion_model = training_latent_diffusion2(encoder, generator, model, train_loader, img_size,
                                                                            epochs, lr, T, device, fox, pretrained = True)

    print("Training model done")

if generate:
    print("Generating sampled dataset ...")


    print("Generating sampled dataset done")