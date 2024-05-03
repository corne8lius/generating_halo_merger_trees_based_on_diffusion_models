import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from torch import nn
import math
from torch.optim import Adam
import tqdm
import argparse

from useful_functions import *
from check_consistency import *
from GAN_model import Encoder, Generator
from diffusion_model import SimpleUnet, Autoencoder_diffusion
from train import * 

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")

    parser.add_argument("--fox", default = True)

    parser.add_argument("--latent", default = False)

    parser.add_argument("--AE", default = False)
    parser.add_argument("--pretrained", default = False)

    parser.add_argument("--train", default = True)
    parser.add_argument("--save", default = False)

    parser.add_argument("--generate", default = True)

    args = parser.parse_args()

    epochs = 1000 # Try more!
    img_size = 64
    batch_size = 256
    lr = 3e-4
    T = 10000
    local_test_num_images = 16
    if not args.fox:
        T = 10
        epochs = 1
        batch_size = 16


    seed_everything(1238)

    print("Loading data ...") 
    if args.fox:
        data_path = "/fp/projects01/ec35/homes/ec-corneb/data/dataset_distlognorm_massnorm.pt"
    else:
        data_path = "../notebooks/data/dataset_distlognorm_massnorm.pt"
    dataset = torch.load(data_path)

    if not args.AE:
        print("Using simple Unet")
        nvar = dataset.shape[1]
        nbr = dataset.shape[3]
        latent_dim = 300
        model = Autoencoder_diffusion(nvar, nbr, latent_dim)
        if False:
            transform = transforms.Resize((img_size, img_size))
            dataset = transform(dataset)
            model = SimpleUnet()

    if args.fox:
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    else:
        dataloader = DataLoader(dataset[:local_test_num_images], batch_size = batch_size, shuffle = True)
    print("Loading data done")

    if args.AE:
        print("Loading model ...")
        if args.pretrained:
            print("\tUsing pretrained Autoencoder")
            if args.fox:
                generator_path = "/fp/projects01/ec35/homes/ec-corneb/models/fox_attempt1/generator_fox_test.pt"
                encoder_path = "/fp/projects01/ec35/homes/ec-corneb/models/fox_attempt1/encoder_fox_test.pt"
            else:
                generator_path = "../notebooks/models/fox_attempt1/generator_fox_test.pt"
                encoder_path = "../notebooks/models/fox_attempt1/encoder_fox_test.pt"

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

        else:
            print("\tUsing an untrained Autoencoder")
            nvar = dataset.shape[1]
            nsnap = dataset.shape[2]
            nbr = dataset.shape[3]
            latent_dim = 300

            encoder = Encoder(nvar, nbr, latent_dim).to(device)
            generator = Generator(nvar, nbr, latent_dim).to(device)

        AE = [encoder, generator]
        model = AE
        print("Loading model done")

    if args.train:
        print("Training model ...")
        model = diffusion_train(model, dataloader, epochs, batch_size, lr, T, device, args)

        if not args.fox and args.AE:
            encoder, decoder = model[0], model[1]
            encoder_name = "../notebooks/diffusion_notebook/autoencoder/encoder.pt"
            decoder_name = "../notebooks/diffusion_notebook/autoencoder/decoder.pt"
            torch.save(decoder.state_dict(), decoder_name)
            torch.save(encoder.state_dict(), encoder_name)
            print(f"saved AE diffusion model")
        print("Training model done")

    if args.generate:
        print("Generating data ...")
        T = 1000
        #model = SimpleUnet()
        model = Autoencoder_diffusion(nvar, nbr, latent_dim)
        if args.fox:
            model_path = "/fp/projects01/ec35/homes/ec-corneb/diffusion/autoencoder/diffusion_test1.pt"
        else:
            model_path = "../notebooks/diffusion_notebook/autoencoder/diffusion_test_AE.pt"

        saved_model = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(saved_model)
        reference_shape = dataset.shape
        img_size = dataset.shape[2:]

        generated = create_generated_images_diffusion(model, img_size, reference_shape, T = T)

        if args.fox:
            save_data_path = "../diffusion/fox1/diffusion_data_AE.pt"
        else:
            save_data_path = "../notebooks/diffusion_notebook/autoencoder/diffusion_data_AE.pt"
        torch.save(generated, save_data_path)
        print(f"saved generated data as", save_data_path)

        print("Generating data done")