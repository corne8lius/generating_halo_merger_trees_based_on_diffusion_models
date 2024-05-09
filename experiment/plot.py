import matplotlib.pyplot as plt
import numpy as np
import torch
import random

def unnormalize(tensor, mini = 0, maxi = 3.2445):
    """
    Unormalize a the distance variable of a tensor that was normalized
    """
    
    return tensor * (maxi - mini) + mini
    
def de_log_normalization(tensor, in_scale = 1000, in_add = 1):
    """
    Undo the log scaling normalization that was applied to the distance variable
    """
    
    return ((10 ** tensor) - in_add) / in_scale

def reset_normalize(tensor):
    """
    Transform the distance variable back to its original value.
    Undo the distance nromalization and scaling done in the preprocessing-
    """
    return de_log_normalization(unnormalize(tensor))

def plot_graph_merger_tree_branch_mass_subhalo_type1(merger_tree):
    """
    Plot a merger tree in terms of branch and snapshot
    """
    image = merger_tree.to(dtype=torch.float64)
    
    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = plt.cm.get_cmap("jet")

    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()

    new_indices = torch.stack([indices[:, 0], indices[:, 1], mass_nonzero, subh_nonzero], dim=0).permute(1, 0)
    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    # Plot the values
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            ax.plot(x[idx] + 1, y[idx], linestyle='-', color='black')

    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            ax.plot([x[i] + 1, 1], [y[i], y[i] + x[i]*0.05], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                ax.plot([x[i+1] + 1, 1], [y[i+1], y[i+1] + x[i+1]*0.05], color='black', linestyle='-')
            else:
                ax.plot([x[i] + 1, 1], [y[i+1], y[i+1] + x[i]*0.05], color='black', linestyle='-')

    ax.scatter(ind05[:, 1] + 1, ind05[:, 0], marker="^", s=50, c=ind05[:, 2], cmap=cmap, zorder=3)
    ax.scatter(ind1[:, 1] + 1, ind1[:, 0], marker="s", s=50, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('Branches')
    ax.set_ylabel('Snapshot')
    ax.set_ylim(29, 0)
    ax.set_title('Sample Plot with Reversed Y-axis')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value")

    return fig, ax

def plot_graph_merger_tree_dist_mass_subhalo_type1(merger_tree, merge=False):
    """
    Plot a merger tree in terms of distance to main branch and snapshot
    """
    image = merger_tree.to(dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(6, 10))
    cmap = plt.cm.get_cmap("jet")

    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()
    dist_nonzero = reset_normalize(dist[dist != 0.0].flatten())

    zero_positions = indices[:, 1] == 0.0

    new_dist = []
    dist_idx = 0
    for idx in zero_positions:
        if idx:
            new_dist.append(0.0)
        else:
            new_dist.append(dist_nonzero[dist_idx])
            dist_idx +=1

    new_indices = torch.stack([indices[:, 0], indices[:, 1],
                               mass_nonzero, subh_nonzero,
                               torch.Tensor(new_dist)],
                              dim=0).permute(1, 0)

    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    dist_nonzero = indices1[:, 4].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            ax.plot(dist_nonzero[idx], y[idx], linestyle='-', color='black')
            
    if merge:
        for i in range(len(x) - 1):
            if x[i] != x[i + 1]:
                ax.plot([dist_nonzero[i], 0], [y[i], y[i] + 1], color='black', linestyle='-')
            if i + 2 == len(x):
                val, count = np.unique(x, return_counts=True)
                if count[-1] == 1:
                    ax.plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')
                else:
                    ax.plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')


    ax.scatter(ind05[:, 4], ind05[:, 0], marker="^", s=50, c=ind05[:, 2], cmap=cmap, zorder=3)
    ax.scatter(ind1[:, 4], ind1[:, 0], marker="s", s=50, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    ax.invert_yaxis()
    
    # Set labels and title
    ax.set_xlabel('Distance from main branch')
    ax.set_ylabel('Snapshot')
    ax.set_ylim(29, 0)
    ax.set_title('Sample Plot with Reversed Y-axis')

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value")
    
    
    return fig, ax

def plot_side_by_side(dataset, save = False, name = "test"):
    """
    Plot three examples of the same merger tree.
        - One is an image representation
        - One is a graphical representation where the y-axis is the time (snapshots) and the x-axis is the branches
        - One is a graphical representation where the y-axis is the time (snapshots) and the x-axis is the distance to the main branch

    Can also save the plot containing the three representations if desired
    """
    
    num_images = len(dataset)
    
    i = random.randint(0, num_images - 1)
    image = dataset[i]
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 10))
    
    
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_xlabel('Branches', fontsize=20)
    axs[0].set_ylabel('Snapshot', fontsize=20)
    axs[0].set_title('Merger Tree image illustration', fontsize=20)
    
    cmap = plt.cm.get_cmap("jet")
    
    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()

    new_indices = torch.stack([indices[:, 0], indices[:, 1], mass_nonzero, subh_nonzero], dim=0).permute(1, 0)
    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    # Plot the values
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            axs[1].plot(x[idx] + 1, y[idx], linestyle='-', color='black')

    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            axs[1].plot([x[i] + 1, 1], [y[i], y[i] + x[i]*0.05], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                axs[1].plot([x[i+1] + 1, 1], [y[i+1], y[i+1] + x[i+1]*0.05], color='black', linestyle='-')
            else:
                axs[1].plot([x[i] + 1, 1], [y[i+1], y[i+1] + x[i]*0.05], color='black', linestyle='-')

    axs[1].scatter(ind05[:, 1] + 1, ind05[:, 0], marker="^", s=100, c=ind05[:, 2], cmap=cmap, zorder=3)
    axs[1].scatter(ind1[:, 1] + 1, ind1[:, 0], marker="s", s=75, c=ind1[:, 2], cmap=cmap, zorder=3)

    # Reverse the y-axis
    axs[1].invert_yaxis()
    
    # Set labels and title
    axs[1].set_xlabel('Branches', fontsize=20)
    axs[1].set_ylabel('Snapshot', fontsize=20)
    axs[1].set_ylim(29, 0)
    axs[1].set_title("Merger Tree - branch", fontsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())
    plt.colorbar(sm, label="mass value", ax=axs[2])
    
    dist = image[0]
    mass = image[1]
    subh = image[2]
    indices = torch.nonzero(subh)
    mass_nonzero = mass[mass != 0.0].flatten()
    subh_nonzero = subh[subh != 0.0].flatten()
    dist_nonzero = reset_normalize(dist[dist != 0.0].flatten())

    zero_positions = indices[:, 1] == 0.0

    new_dist = []
    dist_idx = 0
    for idx in zero_positions:
        if idx:
            new_dist.append(0.0)
        else:
            new_dist.append(dist_nonzero[dist_idx])
            dist_idx +=1

    new_indices = torch.stack([indices[:, 0], indices[:, 1],
                               mass_nonzero, subh_nonzero,
                               torch.Tensor(new_dist)],
                              dim=0).permute(1, 0)

    indices1 = torch.stack(sorted(new_indices, key=lambda new_indices: new_indices[1]))
    y = indices1[:, 0].numpy()
    x = indices1[:, 1].numpy()
    mass_nonzero = indices1[:, 2].numpy()
    subh_nonzero = indices1[:, 3].numpy()
    dist_nonzero = indices1[:, 4].numpy()
    
    num_branches = np.unique(x)

    ind05 = indices1[indices1[:, 3] == 0.5].numpy()
    ind1 = indices1[indices1[:, 3] == 1].numpy()

    
    for yv in np.unique(x):
        if yv != np.nan:
            idx = x == yv
            axs[2].plot(dist_nonzero[idx], y[idx], linestyle='-', color='black')
            
    for i in range(len(x) - 1):
        if x[i] != x[i + 1]:
            axs[2].plot([dist_nonzero[i], 0], [y[i], y[i] + 1], color='black', linestyle='-')
        if i + 2 == len(x):
            val, count = np.unique(x, return_counts=True)
            if count[-1] == 1:
                axs[2].plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')
            else:
                axs[2].plot([dist_nonzero[i+1], 0], [y[i+1], y[i+1] + 1], color='black', linestyle='-')


    axs[2].scatter(ind05[:, 4], ind05[:, 0], marker="^", label ='Sattelite', s=100, c=ind05[:, 2], cmap=cmap, zorder=3)
    axs[2].scatter(ind1[:, 4], ind1[:, 0], marker="s", label ='Main halo', s=75, c=ind1[:, 2], cmap=cmap, zorder=3)


    # Reverse the y-axis
    axs[2].invert_yaxis()
    
    # Set labels and title
    axs[2].set_xlabel('Distance from main branch', fontsize=20)
    axs[2].set_ylabel('Snapshot', fontsize=20)
    axs[2].set_ylim(29, 0)
    axs[2].set_title('Merger Tree - Distance to \n main branch', fontsize=20)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=mass_nonzero.min(), vmax=mass_nonzero.max())

    
    plt.tight_layout()
    plt.legend(title='Markers', loc='upper right')
    if save:
        plt.savefig(f'merger_trees/{name}.png')
        print(f"saved fig {name}")
        
    plt.show()