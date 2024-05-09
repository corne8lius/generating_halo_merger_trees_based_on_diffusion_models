from torchvision import transforms 
import torch
import warnings
import random
import argparse
warnings.filterwarnings('ignore')

from experiment.useful_functions import *
from experiment.evaluation import *
from experiment.diffusion_model import UNet, Diffusion
from experiment.train import * 
from experiment.plot import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")

    parser.add_argument("--generate", default = True)
    parser.add_argument("--random", default = False)
    parser.add_argument("--analyze", default = False)

    args = parser.parse_args()

    T = 1000
    interpolation = transforms.InterpolationMode.BILINEAR

    # load data
    training_data = torch.load("data/dataset_normalized_consistent.pt")
    generated_data = torch.load("diffusion/Cornelius_diffusion/generated_merger_trees/generated_merger_trees.pt", map_location = device)
    diffusion_model = torch.load("diffusion/Cornelius_diffusion/model/diffusion_model.pt", map_location = device)
    model = UNet()
    model.load_state_dict(diffusion_model)

    # generate new sample and evaluate and visualize it
    if args.generate:
        print("\nGenerating new sample ...")
        print(f"Diffusion process need to go through {T} iterations ...")
        t = torch.Tensor([T])
        diffusion = Diffusion(T)
        sample = diffusion.sample(model, 1)
        transform = transforms.Resize((29, 10), interpolation = interpolation)
        sample = transform(sample.to(dtype=torch.float))
        sample = normalize(sample, True, [0, 1, 2])

        sample = transform_diffusion_image(sample, d_thresh = 0.245, m_tresh = 0.5, s_low = 0.3, s_high = 0.72)
        plot_side_by_side(sample)
        full_evaluation(sample, training_data)   

    # draw sample from existing dataset and evaluate and visualize it
    if args.random:
        i = random.randint(0, len(generated_data) - 1)
        sample = generated_data[i].unsqueeze(0)
        plot_side_by_side(sample)
        full_evaluation(sample, training_data)

    # analyze the complete generated merger tree dataset
    if args.analyze:
        full_evaluation(generated_data, training_data)