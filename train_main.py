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
import argparse
warnings.filterwarnings('ignore')

from experiment.useful_functions import *
from experiment.check_consistency import *
from experiment.diffusion_model import UNet, Diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(1238)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")

    parser.add_argument("--train", default = True)
    parser.add_argument("--generate", default = True)

    args = parser.parse_args()


    batch_size = 16
    epochs = 200
    T = 1000
    img_size = 64
    lr = 3e-4
    linear = True
    interpolation_mode = ("BILINEAR", transforms.InterpolationMode.BILINEAR)
    interpolation = interpolation_mode[1]
    interpolation_name = interpolation_mode[0]
    T_name = str(T)


    print("scheudler = Linear")
    print("Interpolation mode = ", interpolation_name)
    print("T = ", T)
    print("Epochs =", epochs)

    name = "model_epochs=" + str(epochs) + "_" + interpolation_name + "_" + "T=" + T_name  + ".pt"

    data_path = "notebooks/data/dataset_normalized_consistent_only.pt"
    dataset = torch.load(data_path)
    transform = transforms.Resize((img_size, img_size), interpolation = interpolation)
    resized_image = transform(dataset)
    train_loader = DataLoader(resized_image, shuffle=True, batch_size = batch_size)

    if args.train:     
        print("Training ...")
        model = UNet()
        diffusion_model = Diffusion(T = T, img_size = img_size, device = device)
        start_time = time.time()
        loss_list, sampled_images, model = train(train_loader, epochs, lr, device, name, model, diffusion_model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Training time:", elapsed_time, "seconds", flush=True)
        print("Training done ...")

    if args.generate:
        print("Generating data ...")
        T = T
        model = UNet()
        diffusion_model = Diffusion(T = T, img_size = img_size, device = device)

        model_path = "notebooks/diffusion_notebook/diffusion_test/" + name

        saved_model = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(saved_model)
        
        reference_shape = dataset.shape

        start_time = time.time()
        generated = create_generated_images_diffusion2(model, img_size, reference_shape, T, interpolation, diffusion_model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Generating 10k images time:", elapsed_time, "seconds", flush=True)

        save_data_path = "notebooks/diffusion_notebook" + name
        torch.save(generated, save_data_path)
        print(f"saved generated data as", save_data_path)

        print("Generating data done")