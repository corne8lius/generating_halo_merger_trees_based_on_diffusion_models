
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_values(tensor, lower_threshold = 0.19, upper_threshold = 0.77):
    """
    Map values of a tensor between 0 and 1 to 0.0, 0.5, or 1.0 based on thresholds.

    Args:
    - tensor: Input tensor with values between 0 and 1.
    - lower_threshold: Lower threshold for mapping (inclusive).
    - upper_threshold: Upper threshold for mapping (exclusive).

    Returns:
    - mapped_tensor: Tensor with values mapped to 0.0, 0.5, or 1.0.
    """

    # Create a tensor of zeros with the same shape as the input tensor
    mapped_tensor = torch.zeros_like(tensor)

    # Map values based on thresholds
    mapped_tensor[[tensor <= lower_threshold]] = 0.0
    mapped_tensor[(tensor > lower_threshold) & (tensor < upper_threshold)] = 0.5
    mapped_tensor[tensor >= upper_threshold] = 1.0

    return mapped_tensor

def save_model(encoder, generator, discriminator,
               optimizer_discriminator, optimizer_enc_dec,
               disc_name, gen_name, enc_name,
               nbr, nsnap, nvar, latent_dim):
    state_dict_disc = {
                        "discriminator": discriminator.state_dict(),
                        "optimizer": optimizer_discriminator.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }


    state_dict_gen = {
                        "generator": generator.state_dict(),
                        "optimizer": optimizer_enc_dec.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }


    state_dict_enc = {
                        "encoder": encoder.state_dict(),
                        "optimizer": optimizer_enc_dec.state_dict(),
                        "latent_space": latent_dim,
                        "num_branches": nbr,
                        "nsnap": nsnap,
                        "nvar": nvar
                        }

    torch.save(state_dict_disc, disc_name)
    print(f"saved discriminator as {disc_name}")
    
    torch.save(state_dict_gen, gen_name)
    print(f"saved generator as {gen_name}")
    
    torch.save(state_dict_enc, enc_name)
    print(f"saved encoder as {enc_name}")

def load_model(discriminator_path, encoder_path, generator_path,
               discriminator_model, encoder_model, generator_model):
    
    # load generator
    saved_generator = torch.load(generator_path, map_location='cpu')
    generator_trained = saved_generator["generator"]
    latent_dim = saved_generator["latent_space"]
    num_branches = saved_generator["num_branches"]
    nsnap = saved_generator["nsnap"]
    nvar = saved_generator["nvar"]
    
    # load encoder
    saved_encoder = torch.load(encoder_path, map_location='cpu')
    encoder_trained = saved_encoder["encoder"]
    
    # load discriminator
    saved_discriminator = torch.load(discriminator_path, map_location='cpu')
    discriminator_trained = saved_discriminator["discriminator"]

    # create generator
    generator = generator_model(nvar, num_branches, latent_dim)
    generator.load_state_dict(generator_trained)
                                
    # create encoder
    encoder = encoder_model(nvar, num_branches, latent_dim)
    encoder.load_state_dict(encoder_trained)
    
    # create discriminator
    discriminator = discriminator_model(nvar, num_branches)
    discriminator.load_state_dict(discriminator_trained)                        
                
                              
    return discriminator, encoder, generator, latent_dim


def create_generated_images(generator, latent_dim, reference_shape, noise_uniform = True):
    generated_images = []
    num_images = reference_shape[0]
    batch_size = 256
    runs = math.floor(num_images / 256)
    extra = num_images % 256
    
    for run in range(runs + 1):
        if run == runs:
            batch_size = extra
            
        if noise_uniform:
            high = 1.0
            low = -1.0
            uniform_noise = torch.rand(batch_size, latent_dim) * (high - (low)) + (low)
            noise = uniform_noise.to(device)
        
        else:
            noise = torch.randn(batch_size, latent_dim).to(device)
            
        fake_images = generator(noise)
        generated_images.append(fake_images)
    
    images = torch.cat(generated_images).detach()
    
    images[:, 2] = map_values(images[:, 2])
    
    print("reconstructed image:")
    i = random.randint(0, num_images - 1)
    img = images[i].permute(1, 2, 0).detach().numpy()
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.show()
    
    print(f"generated {images.shape[0]} images")
    
    return images
    
    


def extract_distribution(images, dim = 0):
    # extract variable
    variable = images[:, dim]
    
    # extract nonzero values
    nonzero_indices = torch.nonzero(variable.flatten())
    nonzero_value = variable.flatten()[nonzero_indices[:, 0]]
    
    # get nonzero distribution
    nonzero_distribution = nonzero_value
    
    # get regular distribution
    distribution = variable.flatten()
    
    # plot nonzero dist
    plt.hist(nonzero_value.flatten().numpy(), bins=100, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('distributions of Nonzero Values')
    plt.show()
    
    # plot regular dist
    plt.hist(distribution.numpy(), bins=100, color='blue')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('distributions of Nonzero Values')
    plt.show()
    print(f"number of nonzero values = {len(nonzero_value)}")
    
    return distribution, nonzero_distribution


def get_large_sample_ks_significance_level(dist1, dist2, alpha):
    m = len(dist1.flatten())
    n = len(dist2.flatten())
    
    sig_level = np.sqrt(-np.log(alpha / 2) * ((1 + (m / n)) / (2 * m)))
    
    return sig_level


    
def ks_test(real_images, fake_images, dim = 0):
    print("Real images:")
    real_dist, real_nonzero_dist = extract_distribution(real_images, dim)
    print("Generated images:")
    fake_dist, fake_nonzero_dist = extract_distribution(fake_images, dim)
    
    plt.hist(real_dist.numpy(), bins=100, alpha=0.5, label='real values', color='blue')
    plt.hist(fake_dist.numpy(), bins=100, alpha=0.5, label='generated values', color='orange')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Full distributions')
    plt.legend()

    # Show plot
    plt.show()
    
    
    statistic, p_value = ks_2samp(real_dist, fake_dist)

    print("Full distribution:")
    # Print the test results
    print("KS Statistic:", statistic)
    print("P-value:", p_value)

    # Interpret the results
    alpha = 0.05  # significance level
    print(f"\nAccording to regular significance level of {alpha}:")
    if p_value > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
        
     # Interpret the results
    sig_level = get_large_sample_ks_significance_level(real_dist, fake_dist, alpha)  # significance level
    print(f"\nAccording to large sample significance level of {alpha}, giving significance level of {sig_level:.4f}:")
    if statistic < sig_level:
        print(f"The distributions are not significantly different (fail to reject H0), KS statistic {statistic:.4f} < {sig_level:.4f}")
    else:
        print(f"The distributions are significantly different (reject H0), KS statistic {statistic:.4f} > {sig_level:.4f}")
       
    
        
    
    plt.hist(real_nonzero_dist.numpy(), bins=100, alpha=0.5, label='real values', color='blue')
    plt.hist(fake_nonzero_dist.numpy(), bins=100, alpha=0.5, label='generated values', color='orange')

    # Add labels and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Nonzero distributions')
    plt.legend()

    # Show plot
    plt.show()
        
    statistic_nonzero, p_value_nonzero = ks_2samp(real_nonzero_dist, fake_nonzero_dist)
    print("Nonzero distribution:")
    # Print the test results
    print("KS Statistic:", statistic_nonzero)
    print("P-value:", p_value_nonzero)

    # Interpret the results
    alpha = 0.05  # significance level
    print(f"\nAccording to regular significance level of {alpha}:")
    if p_value_nonzero > alpha:
        print("The distributions are not significantly different (fail to reject H0)")
    else:
        print("The distributions are significantly different (reject H0)")
        
    # Interpret the results
    sig_level = get_large_sample_ks_significance_level(real_nonzero_dist, fake_nonzero_dist, alpha)  # significance level
    print(f"\nAccording to large sample significance level of {alpha}, giving significance level of {sig_level:.4f}:")
    if statistic_nonzero < sig_level:
        print(f"The distributions are not significantly different (fail to reject H0), KS statistic {statistic_nonzero:.4f} < {sig_level:.4f}")
    else:
        print(f"The distributions are significantly different (reject H0), KS statistic {statistic_nonzero:.4f} > {sig_level:.4f}")
       
    

    return statistic, statistic_nonzero



def plot_images(images, num_images = 3):
    idx = torch.randint(0, images.shape[0] - 1, (num_images, 1))
    merger_trees = []
    for i in idx:
        print(f"image {i.item()}:")
        img = images[i].squeeze(0)
        merger_trees.append(img)
        img = img.permute(1, 2, 0).detach().numpy()
        plt.imshow(img)
        plt.axis('off')  # Optional: Turn off axis ticks and labels
        plt.show()
        
    return merger_trees


def check_zeros_same_index(image):
    # separate variables
    dist = image[:, 0]
    mass = image[:, 1]
    subh = image[:, 2]
    
    # replace all nonzero values with 1
    dist = torch.where(dist != 0, torch.tensor(1), dist)
    mass = torch.where(mass != 0, torch.tensor(1), mass)
    subh = torch.where(subh != 0, torch.tensor(1), subh)
    
    # create new tensor which is the sum of nonzero and zero values
    new_tensor = dist + mass + subh
    
    # check new tensor, main branch should be 0.0 or 2.0 (since distance should be 0.0), other spots should be 0.0 or 3.0
    consistency_sub = torch.logical_or(new_tensor[:, :, 1:] == 0.0, new_tensor[:, :, 1:] == 3.0)
    consistency_main = torch.logical_or(new_tensor[:, :, 0] == 0.0, new_tensor[:, :, 0] == 2.0).unsqueeze(2)
    consistency = torch.cat([consistency_main, consistency_sub], dim = 2)


    equal = consistency.sum() / (image.shape[-2] * image.shape[-1])
    
    return equal

def check_zero_dist_main_branch(image):
    tree = image[0].unsqueeze(0)
    dist = tree[:, 0]
    main_branch = dist[:, :, 0]
    main_branch_zero = sum(main_branch == 0).sum() / len(main_branch.flatten())
    
    return main_branch_zero

def check_gaps_between_branches(image):
    where = 0
    where_chan = 0
    inconsistent = False
    for channel in range(image.shape[1]):
        stop_branch = False
        for bran in range(image.shape[-1]):
            if channel == 0 and bran == 0:
                continue
            branch = image[:, channel,: ,bran]
            if branch.sum() > 0:
                if stop_branch:
                    where = bran
                    where_chan = channel
                    inconsistent = True
                    break
                else:
                    continue
            elif branch.sum() == 0:
                stop_branch = True

        if inconsistent:
            break
            
    return inconsistent, where, where_chan


def check_gaps_in_branch(image):
    where_bran = 0
    where_row = 0
    inconsistent = False
    for channel in range(image.shape[1]):
        for bran in range(image.shape[-1]):
            stop_branch = False
            branch_started = False
            for row in range(image.shape[2]):
                pixel = image[:, channel, row ,bran]
                if pixel == 0 and branch_started == False:
                    continue
                elif pixel == 0 and branch_started:
                    stop_branch = True
                    continue
                elif pixel != 0 and stop_branch == False:
                    branch_started = True
                    continue
                elif pixel != 0 and stop_branch:
                    where_bran = bran
                    where_row = row
                    inconsistent = True
                    break
            if inconsistent:
                break
                
        if inconsistent:
            break
            
    return inconsistent, where_bran, where_row

def check_last_descendant(image):
    mass = image[0, 1, -1, 0] != 0.0
    subh = image[0, 2, -1, 0] != 0.0
    
    exist = mass and subh
    
    no_other_progenitor = image[:, :, -1, 1:].sum().item() == 0.0
    
    last_descendant = exist and no_other_progenitor
    
    
    return last_descendant


def check_branch_length(dataset):
    print("==================================================================================================================================", flush=True)
    print("Complexity:")
    print("==================================================================================================================================", flush=True)
    one_branch = []
    two_branch = []
    three_branch = []
    four_branch = []
    five_branch = []
    six_branch = []
    seven_branch = []
    eight_branch = []
    nine_branch = []
    ten_branch = []

    for datapoint in dataset:
        one_channel = datapoint[1]
        branches = torch.count_nonzero(one_channel, dim=0)
        num_branch = torch.count_nonzero(branches, dim=0)
        if num_branch == 1:
            one_branch.append(datapoint)
        elif num_branch == 2:
            two_branch.append(datapoint)
        elif num_branch == 3:
            three_branch.append(datapoint)
        elif num_branch == 4:
            four_branch.append(datapoint)
        elif num_branch == 5:
            five_branch.append(datapoint)
        elif num_branch == 6:
            six_branch.append(datapoint)
        elif num_branch == 7:
            seven_branch.append(datapoint)
        elif num_branch == 8:
            eight_branch.append(datapoint)
        elif num_branch == 9:
            nine_branch.append(datapoint)
        elif num_branch == 10:
            ten_branch.append(datapoint)

    total = [one_branch, two_branch, three_branch, four_branch, five_branch,
             six_branch, seven_branch, eight_branch, nine_branch, ten_branch]

    total_im = 0
    total_branches = 0
    for i, branch_list in enumerate(total):
        total_im += len(branch_list)
        total_branches += (i + 1) * len(branch_list)
        print(f"number of images with {i + 1} branches is: {len(branch_list)}")

    print(f"\ntotal images = {total_im}")

    avg_branch = total_branches / total_im
    print(f"\naverage number of branches in an image = {avg_branch:.2f} vs. 7.12 in training data")

    nonzero_indices = torch.nonzero(dataset[:, 1].flatten())
    nonzero_value = len(dataset[:, 1].flatten()[nonzero_indices[:, 0]])

    average_nonzero_entries = nonzero_value / total_im 
    avg_branch_length = nonzero_value / total_branches
    print(f"Average branch length = {avg_branch_length:.2f} vs. 9.06 in training data")
    print(f"Average nonzero entries (progenitors) = {average_nonzero_entries:.2f} vs. 64.55 in training data")
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)
    print("\n\n", flush=True)
    

def mass_not_preserved_percentage_decrease(dataset, threshold, original_value):
    branches = dataset.shape[-1]

    total_diffs = []

    not_preserving_mass = 0
    total_progenitors = 0

    max_decrease = threshold


    for im in dataset:
        for i in range(branches - 1):
            branch = im[:, i]
            nonzero_values = branch[branch.nonzero()].squeeze(-1)
            total_progenitors += len(nonzero_values)

            if len(nonzero_values) > 1:
                diffs = nonzero_values[1:] - nonzero_values[:-1]
                percentage_diffs = diffs / nonzero_values[:-1]
                not_preserved = percentage_diffs[percentage_diffs < max_decrease]
                not_preserving_mass += len(not_preserved)
                total_diffs.append(percentage_diffs.flatten().tolist())
                
    print(f"monotonicity threshold = {max_decrease}% change")
    print(f"number of occurences where mass is not preserved = {not_preserving_mass}")
    print(f"perc of occurences where mass is not preserved = {(100 * not_preserving_mass / total_progenitors):.2f}% vs. {original_value}% in training data")
    print("\n")
    return total_diffs

def distance_drastic_jumps(dataset, threshold):
    branches = dataset.shape[-1]

    total_diffs = []
    
    not_preserved_dist = 0
    total_progenitors = 0
    
    total_merger_distance_fail = 0


    for im in dataset:
        for i in range(branches):
            branch = im[:, i]
        
            nonzero_values = branch[branch.nonzero()].squeeze(-1)
            total_progenitors += len(nonzero_values)

            if len(nonzero_values) > 1:
                diffs = nonzero_values[1:] - nonzero_values[:-1]
                percentage_diffs = diffs / nonzero_values[:-1]
                not_preserved = percentage_diffs[percentage_diffs < 0.0]
                not_preserved_dist += len(not_preserved)
                total_diffs.append(percentage_diffs.flatten().tolist())
                
                minimum_branch = nonzero_values.min()
                last = nonzero_values[-1]
                if last != minimum_branch:
                    total_merger_distance_fail += 1
                
    print(f"total distance progentors = {total_progenitors} ")
    print(f"number of occurences where distance increase (not preserved) = {not_preserved_dist}")
    print(f"perc of occurences where mass increase (not preserved) = {(100 * not_preserved_dist / total_progenitors):.2f}% vs. 49.67% in training data")
    print("\n")
    print("total branches where the last halo distance to main branche is not the lowest in the branch is = ", total_merger_distance_fail)
    perc = round(100 * total_merger_distance_fail / (branches * len(dataset)), 2)
    print(f"percentage of all branches where the last halo distance to main branch is not the lowest in the branch is = {perc}% vs. 17.88% in training data")
    print("\n")
    
    
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.tensor(jumps)

    abs_j = torch.abs(j)

    # Boolean indexing to get values above the threshold
    values_above_threshold = abs_j[abs_j > threshold]
    print(f"jumps greater than {100 * threshold}% (negative or positive) = {100 * (len(values_above_threshold) / (total_progenitors)):.2f}% of all jumps vs. 4.76% in training data")

    return total_diffs, total_progenitors

def subhalo_last_type_is_sattelite(dataset):
    branches = dataset.shape[-1]

    not_preserving_subhalo = 0
    total_branches = 0
    for im in dataset:
        for i in range(branches - 1):
            branch = im[:, i + 1]
            nonzero_values = branch[branch.nonzero()].squeeze(-1)
            total_branches += 1

            if len(nonzero_values):
                if nonzero_values[-1] != 0.5:
                    not_preserving_subhalo += 1
                
    print(f"perc of branches where last progenitor type is not subhalo = {(100 * not_preserving_subhalo / total_branches):.2f}% vs 14.52% in training data")
    print("\n")
    return

#=========================================================================================================================
                                            #IMPORTANT
#=========================================================================================================================
def variable_consistancy_check(dataset, printer = True):
    distance = dataset[:, 0, :, 1:]
    mass = dataset[:, 1]
    subhalo = dataset[:, 2]

    mass_thresholds = [-0.000001, -0.01, -0.05, -0.1]
    original_values_mass = [25.77, 11.12, 1.71, 0.32]
    distance_treshold = 0.2
    
    print("==================================================================================================================================", flush=True)
    print("MASS:", flush=True)
    print("==================================================================================================================================", flush=True)
    for i in range(len(mass_thresholds)):
        m_t = mass_thresholds[i]
        original_value = original_values_mass[i]
        total_diffs = mass_not_preserved_percentage_decrease(mass, m_t, original_value)
        print("\n")
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.nonzero(torch.tensor(jumps, dtype = torch.float32))
    if len(j) > 0 and printer:
        plt.hist(j.numpy(), bins = 100)
        plt.show()
        print(f"min mass change = {100 * j.min():.2f}% vs. -22.42% in training data", flush=True)
        print(f"max mass change = {100 * j.max():.2f}% vs. 28.61% in training data", flush=True)
        print(f"average mass change = {100 * torch.tensor(j, dtype = torch.float32).mean():.2f}% vs. 00.29% in training data", flush=True)
            
    
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)
    print("\n\n", flush=True)
    print("==================================================================================================================================", flush=True)
    print("DISTANCE:", flush=True)
    print("==================================================================================================================================", flush=True)
    total_diffs, total_progenitors = distance_drastic_jumps(distance, distance_treshold)
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.nonzero(torch.tensor(jumps, dtype = torch.float32))
    if len(j) > 0 and printer:
        plt.hist(j.numpy(), bins = 100)
        plt.show()
        print(f"min jump = {100 * j.min():.2f} vs. -80.70% in training data", flush=True)
        print(f"max jump = {100 * j.max():.2f} vs. 509.33% in training data", flush=True)
        print(f"average jump = {100 * torch.tensor(j, dtype = torch.float32).mean():.2f} vs. -2.60% in training data", flush=True)
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)
    print("\n\n", flush=True)
    print("==================================================================================================================================", flush=True)
    print("SUBHALO:", flush=True)
    print("==================================================================================================================================", flush=True)
    subhalo_last_type_is_sattelite(subhalo)
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)

def check_consistency(images, print_index = False):
    print("==================================================================================================================================")
    print("Consistency:")
    print("==================================================================================================================================")
    
    consistant_images = []
    inconsistent_images = []
    num_images = images.shape[0]
    
    if print_index:
        print("inconsistent images:")
    
    zero_inconsistency = 0
    dist_main_branch = 0
    gap_between_branch = 0
    gap_in_branch = 0
    last_descendant = 0
    
    two_or_more_inconsistencies = 0
    
    for i, image in enumerate(images):
        image = image.unsqueeze(0)
        # check zeros in same index
        zero_same_index = check_zeros_same_index(image)

        # check distance is 0 in main branch
        main_branch_zero = check_zero_dist_main_branch(image)

        # check no gaps between columns
        inconsistent_branch, where_branch, where_chan= check_gaps_between_branches(image)

        # check no gaps between subhalo in same branch
        inconsistent_gap, where_gap_bran, where_gap_row = check_gaps_in_branch(image)
        
        # check that last descendant exist for image
        last_descendant_exist = check_last_descendant(image)

        # check if image is overall consistant and add to consistant images
        if zero_same_index == 1 and main_branch_zero == 1 and not inconsistent_branch and not inconsistent_gap and last_descendant_exist:
            consistant_images.append(image)

        else:
            inc = 0
            inconsistent_images.append(image)
            if print_index:
                if zero_same_index != 1:
                    print(f" image index = {i}, inconsistent because zero not same index: zero in {(100 * zero_same_index):.2f}% of same spots across channels")
                    zero_inconsistency += 1
                    inc += 1
                if main_branch_zero != 1:
                    print(f" image index = {i}, inconsistent because dist in main branch is not zero: zero in {(100 * main_branch_zero):.2f}% of distance main branch")
                    dist_main_branch += 1
                    inc += 1
                if inconsistent_branch:
                    print(f" image index = {i}, inconsistent because gaps between branches {where_branch - 1} and {where_branch + 1} in channel {where_chan}")
                    gap_between_branch += 1
                    inc += 1
                if inconsistent_gap:
                    print(f" image index = {i}, inconsistent because gaps in branch {where_gap_bran} at row {where_gap_row + 1}")
                    gap_in_branch += 1
                    inc += 1
                if not last_descendant_exist:
                    print(f" image index = {i}, inconsistent it does not have a last descendant")
                    last_descendant += 1
                    inc += 1
            
            else:
                if zero_same_index != 1:
                    zero_inconsistency += 1
                    inc += 1
                if main_branch_zero != 1:
                    dist_main_branch += 1
                    inc += 1
                if inconsistent_branch:
                    gap_between_branch += 1
                    inc += 1
                if inconsistent_gap:
                    gap_in_branch += 1
                    inc += 1
                if not last_descendant_exist:
                    last_descendant += 1
                    inc += 1
            
            if inc > 1:
                two_or_more_inconsistencies +=1
    
    if len(consistant_images) != 0:
        consistant_images = torch.stack(consistant_images).squeeze(1)
        
        perc = 100 * consistant_images.shape[0] / num_images
    
        print("\n")
        print(f"Percentage of consistant images = {perc:.2f}% vs. 97.51% in training data")
        
    if len(consistant_images) == 0:
        print("\n")
        print(f"Percentage of consistant images = {0.0}%")
    
    if len(inconsistent_images) != 0:
        inconsistent_images = torch.stack(inconsistent_images).squeeze(1)
        
        inconsistant = len(inconsistent_images)
        print("\nInconsistency reasons:")
        print(f"inconsistency due to zero / nonzero mistake = {(zero_inconsistency * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to distance not zero in main branch =  {(dist_main_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to gap between branches =  {(gap_between_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to zgap in branch {(gap_in_branch * 100 / inconsistant):.2f}%")
        print(f"inconsistency due to last descendant dont exist {(last_descendant * 100 / inconsistant):.2f}%")

        print(f"\nNumber of images with two or more inconsistencies = {two_or_more_inconsistencies}, which is  {(two_or_more_inconsistencies * 100 / inconsistant):.2f}%")
        print(f"That corresponds to {(two_or_more_inconsistencies * 100 / num_images):.2f}% of all images")
        
        print("\n")
        print(f"Of all images, {(zero_inconsistency * 100 / num_images):.2f}% have zero inconsistency")
        print(f"Of all images, {(dist_main_branch * 100 / num_images):.2f}% have distance main branch inconsistency")
        print(f"Of all images, {(gap_between_branch * 100 / num_images):.2f}% have gap between branches inconsistency")
        print(f"Of all images, {(gap_in_branch * 100 / num_images):.2f}% have gap within branch inconsistency")
        print(f"Of all images, {(last_descendant * 100 / num_images):.2f}% have last descendant inconsistency")
    
    print("==================================================================================================================================", flush=True)
    print("==================================================================================================================================", flush=True)
    print("\n\n", flush=True)
    
    return consistant_images, inconsistent_images

def analyze_data(generated_data_path, comparing_data_path):
    generated_data = torch.load(generated_data_path, map_location = "cpu")
    original_data = torch.load(comparing_data_path)
    if original_data.shape[0] == 3:
        original_data = original_data.permute(1, 0, 2, 3)

    images = plot_images(generated_data, 3)

    consistant, inconsistant = check_consistency(generated_data)

    variable_consistancy_check(generated_data, False)

    #avg_branch = check_branch_length(generated_data)
    
    if len(consistant) != 0:
        print("\nCheck number of branches of consistent generated images:")
        check_branch_length(consistant)
    
    ks, ks_nonzero = ks_test(original_data, generated_data, dim = 1)
    
    return generated_data, consistant, inconsistant