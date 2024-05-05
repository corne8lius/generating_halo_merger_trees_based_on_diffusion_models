# Generating halo merger trees based on diffusion models
### overview and information regarding code repository for the master thesis of Cornelius Bencsik in data science at University of Oslo. 
The master thesis evolves around generative models for halo merger tree generation, how to generate and evaluate generated merger trees.

This repository contains all work, analysis, functions, files, data, model, and results from experiments regarding the thesis.

## Folders
### tests – NOT IMPORTANT
The tests folder contains various different scripts and files containing all different experiments that have been completed throughout the thesis. 
These files are fairly messy and might be hard to follow. However, the important files are elsewhere and will be covered, but if there are some experiments or 
different things that are not found elsewhere, it will be in this folder. But it is highly unlikely that it will be anything of importance.

### notebooks – work, analysis and results
The notebooks folder contains many different notebooks that covers all work, analysis, results, and other stuff that has been reported in the thesis. 
The notebooks are referred to in the thesis as for example **“notebook X”**, where X is the number of the notebook. This notebook will be called “*** X – notebook description”
in the notebook folder. Use the notebook folder and files for evidence of work, results and analysis.

### data
The data folder contains the normalized and consistent training data. This is the regular training data that is preprossesed according to the scaling, normalization and 
preprocessing method presented in the thesis. This simply means the distance variable is log scaled and minmax normalized and the mass variable is minmax normalized. 
Then the consistent training merger trees are extracted, leaving a training dataset of 27 601 training merger trees with 5 to 10 branches.

The data has shape (27 601, 3, 29, 10), where the first channel is the distance variable, the second channel is the mass variable and the thirs channel is the subhalo
channel. To use the data, simply copy this code after cloning the repo:
```
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("path_to_repository_folder/data/dataset_normalized_consistent.pt", map_location = device)
```
where you replace **path_to_repository_folder** with the path leading to the cloned repository.

### diffusion
The diffusion folder contains two subfolders: **Cornelius_diffusion** and **diffusion_test**.
#####diffusion_test
The diffusion_test folder is for the model and generated merger tree you create and generate through the reproduction possibility of the project. This is done through **train_main.py**. 
If you were to reproduce the results or create a new model, the model will be saved in **diffusion/diffusion_test/model/** and the generated merger trees will be saved in
**diffusion/diffusion_test/generated_merger_trees/**.

#####Cornelius_diffusion
The Cornelius_diffusion folder contains the existing model and a consistent generated merger trees dataset that is created by me in the thesis. The Cornelius_diffusion folder contains a model subfolder where
the model is stored and a subfolder called generated_merger_trees where the generated merger trees are stored. The model and generated merger trees are already pre-loaded to the reproduction
and analysis scripts in the folder with the correct paths.

you can load the model like this:
```
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("path_to_repository_folder/diffusion/Cornelius_diffusion/model/diffusion_model.pt", map_location = device)
```
where you replace **path_to_repository_folder** with the path leading to the cloned repository. now **model** will contain the state dict to the U-Net model. To generate a new sample,
analyzing and visualizing it, use the following code:
```
from torchvision import transforms
from experiment.diffusion_model import UNet, Diffusion
from experiment.useful_functions import *
from experiment.plot import *
from experiment.evaluation import *

T = 1000
diffusion = Diffusion(T)
sample = diffusion.sample(model, 1)
transform = transforms.Resize((29, 10), interpolation = interpolation)
sample = transform(sample.to(dtype=torch.float))
sample = normalize(sample, True, [0, 1, 2])

sample = transform_diffusion_image(sample, d_thresh = 0.245, m_tresh = 0.5, s_low = 0.3, s_high = 0.72)
plot_side_by_side(sample)
full_evaluation(sample, training_data)   
```

To load the generated merger tree dataset use the following code:
```
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generated_merger_trees = torch.load("path_to_repository_folder/diffusion/Cornelius_diffusion/generated_merger_trees/generated_merger_trees.pt", map_location = device)
```
where you replace **path_to_repository_folder** with the path leading to the cloned repository.

### experiment – IMPORTANT
The experiment folder contains all the useful functions for analyzing, plotting, training, generating merger trees, and the diffusion model. 

Some of the key functions in the folder are the following:
1.	**full_evaluation in evaluation.py:** takes a dataset of merger trees or a single merger tree of the shape (number of trees, 3, number of snapshots, number of branches) and completely evaluate it. If an original dataset is given, the mass ks statistic will also be evaluated. This function is a great contribution toward fully evaluating generated merger trees
2.	**plot_side_by_side in plot.py:** randomly sample a merger tree from a given merger tree dataset and plot it in three different ways.
For more details about the functions, see the functions in the scripts.

## Files
### Main_notebook.ipynb
The main notebook is the main tool for evaluating and visualizing the generated merger trees in different ways. The script contains 6 different code snippets that can be easily run. The code snippets do the following:
1.	Randomly sample a generated merger tree from the consistent generated merger tree dataset in **diffusion/Cornelius_diffusion/generated_merger_trees/** and analyze and visualize it.
2.	Does the same as (1), but given the number of branches you want the merger tree to contain. Simply change the **num_branches** variable to the desired number of branches you want the sampled merger tree to contain.
3.	Does the same as (2) but with complexity instead of branches. Changing the two variables **higher_than** and **complexity** will randomly draw a sample from the generated merger tree dataset with the desired complexity. **higher_than** is a True/False variable where True means you want a higher complexity than the given complexity and False means you want a lower complexity, and **complexity** ranges from around 30 to 140. Sampling a complex merger tree would look like this:
```
higher_than = True
complexity = 120
```
Sampling a sparse merger tree would look like this:
```
higher_than = False
complexity = 40
```
4.	Combines (2) and (3), meaning you can draw a generated merger tree depending on the desired number of branches and complexity. It would be interesting to try different combinations of extreme values meaning few branches - high complexity, few branches - low complexity, many branches - high complexity and many branches – low complexity. There are three variables to change to the desired values. 
Sampling a generated merger tree with many branches and high complexity would look like this:
```
num_branches = 10
higher_than = True
complexity = 135
```
5.	Use the diffusion model in **diffusion/Cornelius_diffusion/model/** to generate a completely new merger tree, visualize and analyze it.
6.	This code snippet deviates from (1) – (5) and analyze and evaluate the whole generated merger tree dataset stored in **diffusion/Cornelius_diffusion/generated_merger_trees/**.
Note that plotting/visualizing a merger tree can give an error. This will only affect one out of the three plots. The function creating that plot sometimes cannot handle the randomly sampled merger tree. The sampled merger tree will still be visualized, but only by the two first plots. If this happens, run the code snippet again and it should draw a new merger tree and visualize it in all three different ways.



### analyze_main.py
**Analyze_main.py** does almost the same things as **main_notebook.ipynb**, but has less options. So for analyzing and visual representation, I would suggest using the notebook.
**Analyze_main.py** has three options:
1. **generate:** generate a completely new sample, visualize and evaluate it.
2. **random:** sample a random merger tree from the generated merger tree dataset, visualize and evaluate it.
3. **analyze:** analyze the full generated merger tree
   
Running the script can simply be done by typing the following in the command line from the cloned folder:
```
$ python3 analyze_main.py
```
The default is to generate a completely new sample (1).

To a´modify the script to do the other experiments, do the following:
```
$ python3 analyze_main.py --generate False --random False --analyze False
```
where you replace **False** with **True** for the experiment you want to do.


### train_main.py - reproduction
**train_main.py** is the main script fo reproduction. The script has three main options:
1. **train:** train a completely new diffusion model.
2. **generate:** generate a completely new merger tree dataset of 10 000 merger trees.
3. **analyze:** analyze and evaluate the generated merger tree dataset.
The default setting is that all of the above options are done in cronological order. But they can be completed separately. The only thing to note is that (2) requires a new diffusion model
to generate new merger trees, which indicates that (1) must have been completed at least once before running (2) and (3) requires generated merger trees to be analyzed, which implies that (2)
must be completed at least once before running (3). My advice is to run the default settings first.

To run the script, do the following:
```
$ python3 train_main.py
```
and modify which experiment you want to run by changing **False** to **True** in the following code:
```
$ python3 train_main.py --train False --generate False --analyze False
```

(1) applies the consistent merger tree dataset in **data** to train a new diffusion model with the following hyperparameters which can be altered:
1. T = 1000                  --T
2. batch_size = 16           --batch_size
3. epochs = 200              --epochs
4. lr = 3e-4                 --lr
I would suggest that the two main hyperparameters to change are T and epochs. The trained model will be stored in **diffusion/diffusion_test/model/** as **"model_epochs=X1_T=X2.pt"**,
where X1 is the number of epochs chosen and X2 is the T value.

(2) requires a trained diffusion model, and the generated merger trees will be saved in **diffusion/diffusion_test/generated_merger_trees/** as **"generated_merger_trees/model_epochs=X1_T=X2.pt"**.
To generate new merger tree from an existing model, make sure the hyperparameter settings are the same as the existing model. However, I suggest to train, generate and analyze in one go, for simplicity-














   
![image](https://github.com/corne8lius/generating_halo_merger_trees_based_on_diffusion_models/assets/168091922/55a9c0bd-8293-4f68-9b0c-12a8e6d487d5)
