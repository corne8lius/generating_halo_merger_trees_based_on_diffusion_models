import torch
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def check_branch_length(dataset, printer = True, print_full = True):
    """
    separate the dataset into subdatasets given the number of branches

    count the merger trees given the number of branches they contain

    produces the complexity metrics:
        - average number of branches
        - average branch length
        - average number of nonzero entries (progenitors)
    """
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
        if printer and print_full:
            print(f"number of images with {i + 1} branches is: {len(branch_list)}")

    avg_branch = total_branches / total_im
    if printer:
        print(f"\naverage number of branches in the image = {avg_branch:.2f}")
    
    nonzero_indices = torch.nonzero(dataset[:, 1].flatten())
    nonzero_value = len(dataset[:, 1].flatten()[nonzero_indices[:, 0]])

    average_nonzero_entries = nonzero_value / total_im 
    avg_branch_length = nonzero_value / total_branches
    if printer:
        print(f"Average branch length = {avg_branch_length:.2f}")
        print(f"Number of nonzero entries (progenitors) = {average_nonzero_entries:.2f}")
        print("\n\n", flush=True)

    return total


def check_zeros_same_index(image):
    """
    check if all three channels have zero and nonzero in the same spots
    """

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
    """
    verify that the distance values in the main branch are all zero
    """
    tree = image[0].unsqueeze(0)
    dist = tree[:, 0]
    main_branch = dist[:, :, 0]
    main_branch_zero = sum(main_branch == 0).sum() / len(main_branch.flatten())
    
    return main_branch_zero

def check_gaps_between_branches(image):
    """
    check that there are no gaps between branches
    """
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
    """
    verify that there are no gaps within branches
    """
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
    """
    verify that the last descendant exist and that it is the only progenitor in the last snapshot
    """
    mass = image[0, 1, -1, 0] != 0.0
    subh = image[0, 2, -1, 0] != 0.0
    
    exist = mass and subh
    
    return exist
    
    
def check_consistency(images, print_index = False):
    """
    Check that all structural consistency requirements are fullfilled

    returns the consistent and inconsistent merger trees separately
    """
    
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
        print(f"Percentage of consistant images = {perc:.2f}%")
        print("\n")
        
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
    
    return consistant_images, inconsistent_images


def mass_not_preserved_percentage_decrease(dataset, threshold):
    """
    check how many mass "jumps" are decreasing in terms of a percentage threshold
    """
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
                
    print(f"Monotonicity threshold = {max_decrease}% change", flush=True)
    print(f"Perc of occurences where mass is not preserved = {(100 * not_preserving_mass / total_progenitors):.2f}%", flush=True)
    return total_diffs

def distance_drastic_jumps(dataset, threshold):
    """
    check how many distance "jumps" increase (not preserved)

    check how many distance "jumps" are drastic in terms changing in terms of a percentage threshold

    check how many of the last distance progenitors in a branch is the lowest in the branch
    """
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
                
    print(f"Percentage of occurences where distance increase (not preserved) = {(100 * not_preserved_dist / total_progenitors):.2f}%", flush=True)
    print("\n", flush=True)
    perc = round(100 * total_merger_distance_fail / (branches * len(dataset)), 2)
    print(f"Percentage of last halo distance is not the lowest in the branch = {perc}% ", flush=True)
    print("\n", flush=True)
    
    
    jumps = [item for sublist in total_diffs for item in sublist]   
    j = torch.tensor(jumps)

    abs_j = torch.abs(j)

    # Boolean indexing to get values above the threshold
    values_above_threshold = abs_j[abs_j > threshold]
    print(f"jumps greater than {100 * threshold}% (negative or positive) = {100 * (len(values_above_threshold) / (total_progenitors)):.2f}% of all jumps", flush=True)

    return total_diffs, total_progenitors

def subhalo_last_type_is_sattelite(dataset):
    """
    check how many of the last progenitors in a branch is of type subhalo
    """
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
                
    print(f"perc of branches where last progenitor type is not subhalo = {(100 * not_preserving_subhalo / total_branches):.2f}%")
    return

def variable_consistancy_check(dataset):
    """
    verify all variable consistencies in one go
    """
    distance = dataset[:, 0, :, 1:]
    mass = dataset[:, 1]
    subhalo = dataset[:, 2]

    mass_thresholds = [-0.000001, -0.1]
    distance_treshold = 0.2
    
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    print("MASS:", flush=True)
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    for i in range(len(mass_thresholds)):
        m_t = mass_thresholds[i]
        total_diffs = mass_not_preserved_percentage_decrease(mass, m_t)
        if i == 0:
            print("\n")   
    
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    print("DISTANCE:", flush=True)
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    total_diffs, total_progenitors = distance_drastic_jumps(distance, distance_treshold)
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    print("SUBHALO:", flush=True)
    print("-------------------------------------------------------------------------------------------------------", flush=True)
    subhalo_last_type_is_sattelite(subhalo)
    print("-------------------------------------------------------------------------------------------------------", flush=True)


def full_evaluation(generated_data, training_data = None):
    """
    Perform a full evaluation of a dataset meaning:
        - structural consistency check
        - complexity check
        - variable consistency check
        - mass KS statistic if a reference dataset is given.
    """
    print("Analyzing generated merger tree dataset ...\n")
    print("=======================================================================================================", flush=True)
    print("\t\t\t CONSISTENCY CHECK")
    print("=======================================================================================================", flush=True)
    consistent, inconsistent = check_consistency(generated_data)
    print("\n=======================================================================================================", flush=True)
    if len(consistent) > 0:
        print("\t\t\t VARIABLE CONSISTENY CHECK of consistent trees")
        print("=======================================================================================================", flush=True)
        variable_consistancy_check(consistent)
        print("\n=======================================================================================================", flush=True)
        if training_data is not None:
            print("\t\t\t MASS KS STATISTIC CHECK of consistent trees")
            print("=======================================================================================================", flush=True)
            statistic, p_value = ks_2samp(training_data[:, 1].flatten(), consistent[:, 1].flatten())
            print(f"\nMass KS Statistic: {statistic:.4f}")
            print("\n=======================================================================================================", flush=True)
        print("\t\t\t COMPLEXIY CHECK of consistent trees")
        print("=======================================================================================================\n", flush=True)
        avg_branch = check_branch_length(consistent)
        print("=======================================================================================================", flush=True)
        print("=======================================================================================================", flush=True)
    print("\nAnalyzing generated merger tree dataset done")


