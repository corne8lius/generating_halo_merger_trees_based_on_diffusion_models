from torchvision import transforms 
from torch.utils.data import DataLoader
import torch
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

from experiment.useful_functions import *
from experiment.evaluation import *
from experiment.diffusion_model import UNet, Diffusion
from experiment.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(1238)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train model")

    parser.add_argument("--train", default = True)
    parser.add_argument("--generate", default = True)
    parser.add_argument("--analyze", default = True)

    parser.add_argument("--T", default = 1000)
    parser.add_argument("--batch_size", default = 16)
    parser.add_argument("--epochs", default = 200)
    parser.add_argument("--lr", default = 3e-4)

    args = parser.parse_args()

    #hyperparameters
    batch_size = args.batch_size
    epochs = args.epochs
    T = args.T
    lr = args.lr
    img_size = 64
    interpolation_mode = ("BILINEAR", transforms.InterpolationMode.BILINEAR)
    interpolation = interpolation_mode[1]
    interpolation_name = interpolation_mode[0]
    T_name = str(T)

    # print hyperparameters
    print("Scheudler = Linear")
    print("Interpolation mode = ", interpolation_name)
    print("T = ", T)
    print("Epochs =", epochs)

    # create model name
    name = "model_epochs=" + str(epochs) + "_" + "T=" + T_name  + ".pt"

    # load and transform trainingdata to desired input shape
    data_path = "data/dataset_normalized_consistent.pt"
    dataset = torch.load(data_path)
    transform = transforms.Resize((img_size, img_size), interpolation = interpolation)
    resized_image = transform(dataset)
    train_loader = DataLoader(resized_image, shuffle=True, batch_size = batch_size)

    # train the model
    if args.train:   
        print("\n=======================================================================================================", flush=True)  
        print("\t\t\t Training ...")
        print("=======================================================================================================", flush=True)
        model = UNet()
        diffusion_model = Diffusion(T = T, img_size = img_size, device = device)
        start_time = time.time()
        loss_list, sampled_images, model = train(train_loader, model, diffusion_model, epochs, T, img_size, lr, name, device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTraining time: {elapsed_time:.4f} seconds", flush=True)
        print("Training done ...")

    # generate new merger trees
    if args.generate:
        print("\n=======================================================================================================", flush=True)
        print("\t\t\t Generating data ...")
        print("=======================================================================================================", flush=True)
        T = T
        model = UNet()
        diffusion_model = Diffusion(T = T, img_size = img_size, device = device)

        model_path = "diffusion/diffusion_test/model/" + name

        saved_model = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(saved_model)
        
        reference_shape = dataset[:10000].shape

        start_time = time.time()
        generated = create_generated_images_diffusion(model, img_size, reference_shape, T, interpolation, diffusion_model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nGenerating 10k images time: {elapsed_time:.4f} seconds", flush=True)

        save_data_path = "diffusion/diffusion_test/generated_merger_trees/" + name
        torch.save(generated, save_data_path)
        print(f"saved generated data as", save_data_path)

        print("Generating data done\n")
        print("\n=======================================================================================================", flush=True)
        
        # analyze generated merger trees
        if args.analyze:
            full_evaluation(generated, dataset)