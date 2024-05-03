import torch
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import argparse
import time

from useful_functions import *
from check_consistency import *
from GAN_model import Discriminator, Encoder, Generator
from train import train_GAN

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")
    #"""
    parser.add_argument("--fox", default = True)

    parser.add_argument("--full_data", default = True)

    parser.add_argument("--diffusion", default = False)

    parser.add_argument("--save", default = False)

    parser.add_argument("--train", default = True)

    parser.add_argument("--generate", default = True)


    args = parser.parse_args()
    print("GAN")
    print(f"FOX = {args.fox}")

    seed_everything(1238)

    print("Loading data ...", flush=True)
 
    if args.fox:
        if args.full_data:
            data_path = "/fp/projects01/ec35/homes/ec-corneb/data/dataset_distlognorm_massnorm.pt"
        else:
            data_path = "/fp/projects01/ec35/homes/ec-corneb/data/ten_branches_norm.pt"
    else:
        if args.full_data:
            data_path = "../notebooks/data/dataset_distlognorm_massnorm.pt"
        else:
            data_path = "../notebooks/data/six_branch_norm.pt"

    dataset = torch.load(data_path)

    print("Loading data done", flush=True)


    if args.train:
        if args.diffusion:
            print("diffusion not implemented", flush=True)
        else:
            print("training GAN ...", flush=True)
            print("test full dataset model ...", flush=True)
            num_train_gen = 1
            noise_uniform = True
            recon_scale = 10
            num_epochs = 200
            batch_size = 256
            latent_dim = 300
            lr = 3e-4
            start_time = time.time()
            rec_image_full_dataset = train_GAN(dataset, Discriminator, Encoder, Generator,
                                                num_train_gen, noise_uniform, recon_scale,args,
                                                num_epochs, batch_size, latent_dim, lr)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Training time:", elapsed_time, "seconds", flush=True)

            print("training GAN done", flush=True)
    if args.generate:
        print("Generating data ...", flush=True)
        if args.fox:
            epochs = [100, 200, 300, 400, 499]
            for epoch in epochs:
                discriminator_path =  f"../models/10_branches/discriminator_retrain_epoch{epoch}.pt"
                encoder_path =  f"../models/10_branches/encoder_retrain_epoch{epoch}.pt"
                generator_path =  f"../models/10_branches/generator_retrain_epoch{epoch}.pt"

                discriminator, encoder, generator, latent_dim = load_model(discriminator_path, encoder_path, generator_path,
                                                                        Discriminator, Encoder, Generator)
                
                reference_shape = dataset[:10000].shape
                start_time = time.time()
                full_dataset_generated = create_generated_images(generator, latent_dim, reference_shape, noise_uniform = True)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Generating 10k images time:", elapsed_time, "seconds", flush=True)

                save_data_path = f"../models/10_branches/generated_images_epoch{epoch}.pt"
                torch.save(full_dataset_generated, save_data_path)
                print(f"saved generated data as", save_data_path, flush=True)

                checked_generated = analyze_data(save_data_path, data_path)

        else:
            discriminator_path =  "../notebooks/models/fox_attempt3/discriminator_fox_original_data_sigmoid.pt"
            encoder_path =  "../notebooks/models/fox_attempt3/encoder_fox_original_data_sigmoid.pt"
            generator_path =  "../notebooks/models/fox_attempt3/generator_fox_original_data_sigmoid.pt"

            discriminator, encoder, generator, latent_dim = load_model(discriminator_path, encoder_path, generator_path,
                                                                    Discriminator, Encoder, Generator)
            
            reference_shape = dataset.shape
            full_dataset_generated = create_generated_images(generator, latent_dim, reference_shape, noise_uniform = True)

            save_data_path = "../notebooks/models/fox_attempt3/generated_images_original_data_sigmoid.pt"
            torch.save(full_dataset_generated, save_data_path)
            print(f"saved generated data as", save_data_path, flush=True)
        print("Generating data done", flush=True)

        
