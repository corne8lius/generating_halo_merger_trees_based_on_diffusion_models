from torchvision import transforms 
import torch
from torchvision import transforms
import warnings
import random
import argparse
warnings.filterwarnings('ignore')

from experiment.useful_functions import *
from experiment.check_consistency import *
from experiment.diffusion_model import UNet, Diffusion
from experiment.train import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")

    parser.add_argument("--diffusion", default = True)
    parser.add_argument("--generate_sample", default = False)
    parser.add_argument("--pick_random_sample", default = False)
    parser.add_argument("--analyze_full_generated_dataset", default = True)

    args = parser.parse_args()

    T = 1000
    interpolation = transforms.InterpolationMode.BILINEAR

    training_data = torch.load("notebooks/data/dataset_normalized_consistent_only.pt")
    generated_data = torch.load("notebooks/diffusion_notebook/diffusion_2.0/diffusion2.0_consistant_postporcessed_images.pt", map_location = device)
    diffusion_model = torch.load("notebooks/diffusion_notebook/diffusion_2.0/diffusion_new_model2.0.pt", map_location = device)
    model = UNet()
    model.load_state_dict(diffusion_model)

    if args.generate_sample:
        print("Generating new sample ...")
        t = torch.Tensor([T])
        diffusion = Diffusion(T)
        sample = diffusion.sample(model, 1)
        transform = transforms.Resize((29, 10), interpolation = interpolation)
        sample = transform(sample.to(dtype=torch.float))
        sample = normalize(sample, True, [0, 1, 2])

        sample = transform_diffusion_image(sample, d_thresh = 0.245, m_tresh = 0.5, s_low = 0.3, s_high = 0.72)
        plot_side_by_side(sample)
        consistent, inconsistent = check_consistency(sample)
        variable_consistancy_check(sample)
        avg_branch = check_branch_length(sample)     

    if args.pick_random_sample:
        i = random.randint(0, len(generated_data) - 1)
        sample = generated_data[i].unsqueeze(0)
        plot_side_by_side(sample)
        consistent, inconsistent = check_consistency(sample)
        variable_consistancy_check(sample)
        avg_branch = check_branch_length(sample)

    if args.analyze_full_generated_dataset:
        print("Analyzing generated merger tree dataset ...")
        consistent, inconsistent = check_consistency(generated_data)
        variable_consistancy_check(generated_data)
        avg_branch = check_branch_length(generated_data)
        ks_test(training_data, generated_data, dim = 1)



