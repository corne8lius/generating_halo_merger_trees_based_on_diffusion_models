import torch
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt


def check_branch_length(dataset):
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

    print(f"total images = {total_im}")

    avg_branch = total_branches / total_im
    print(f"average number of branches in an image = {avg_branch:.2f} vs. 7.12 in training data")

    nonzero_indices = torch.nonzero(dataset[:, 1].flatten())
    nonzero_value = len(dataset[:, 1].flatten()[nonzero_indices[:, 0]])

    average_nonzero_entries = nonzero_value / total_im 
    avg_branch_length = nonzero_value / total_branches
    print(f"Average branch length = {avg_branch_length:.2f} vs. 9.06 in training data")
    print(f"Average nonzero entries (progenitors) = {average_nonzero_entries:.2f} vs. 64.55 in training data")



def plot_images(images, num_images):
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
    dist = image[:, 0]
    mass = image[:, 1]
    subh = image[:, 2]

    same_zero_indices = (dist == 0) & (mass == 0) & (subh == 0)

    equal = same_zero_indices.sum() / (subh == 0).sum()
    
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
    
    return exist
    
    
def check_consistency(images, print_index = False):
    
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
    
    return consistant_images, inconsistent_images


def check_branch_length(dataset):
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

    print(f"total images = {total_im}")

    avg_branch = total_branches / total_im
    print(f"average number of branches in an image = {avg_branch:.2f} vs. 7.12 in training data")

    nonzero_indices = torch.nonzero(dataset[:, 1].flatten())
    nonzero_value = len(dataset[:, 1].flatten()[nonzero_indices[:, 0]])

    average_nonzero_entries = nonzero_value / total_im 
    avg_branch_length = nonzero_value / total_branches
    print(f"Average branch length = {avg_branch_length:.2f} vs. 9.06 in training data")
    print(f"Average nonzero entries (progenitors) = {average_nonzero_entries:.2f} vs. 64.55 in training data")
    