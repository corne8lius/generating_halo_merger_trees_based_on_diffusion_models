\chapter{Appendix D --- Reproduction}\label(reproduction)
\section(Github repository)

\url{https://github.com/corne8lius/generating_halo_merger_trees_based_on_diffusion_models/tree/main}

\section(README.md)
\hypertarget{generating-halo_merger-trees-based-on-diffusion-models}{%
\section{Generating halo\_merger trees based on diffusion
models}\label{generating-halo_merger-trees-based-on-diffusion-models}}

\hypertarget{overview-and-information-regarding-code-repository-for-the-master-thesis-of-cornelius-bencsik-in-data-science-at-university-of-oslo.}{%
\subsubsection{overview and information regarding code repository for
the master thesis of Cornelius Bencsik in data science at University of
Oslo.}\label{overview-and-information-regarding-code-repository-for-the-master-thesis-of-cornelius-bencsik-in-data-science-at-university-of-oslo.}}

The master thesis evolves around generative models for halo merger tree
generation, how to generate and evaluate generated merger trees.

This repository contains all work, analysis, functions, files, data,
model, and results from experiments regarding the thesis.

\hypertarget{folders}{%
\subsection{Folders}\label{folders}}

\hypertarget{tests-not-important}{%
\subsubsection{tests -- NOT IMPORTANT}\label{tests-not-important}}

The tests folder contains various different scripts and files containing
all different experiments that have been completed throughout the
thesis. These files are fairly messy and might be hard to follow.
However, the important files are elsewhere and will be covered, but if
there are some experiments or different things that are not found
elsewhere, it will be in this folder. But it is highly unlikely that it
will be anything of importance.

\hypertarget{notebooks-work-analysis-and-results}{%
\subsubsection{notebooks -- work, analysis and
results}\label{notebooks-work-analysis-and-results}}

The notebooks folder contains many different notebooks that covers all
work, analysis, results, and other stuff that has been reported in the
thesis. The notebooks are referred to in the thesis as for example
\textbf{``notebook X''}, where X is the number of the notebook. This
notebook will be called ``*** X -- notebook description'' in the
notebook folder. Use the notebook folder and files for evidence of work,
results and analysis.

\hypertarget{data}{%
\subsubsection{data}\label{data}}

The data folder contains the normalized and consistent training data.
This is the regular training data that is preprossesed according to the
scaling, normalization and preprocessing method presented in the thesis.
This simply means the distance variable is log scaled and minmax
normalized and the mass variable is minmax normalized. Then the
consistent training merger trees are extracted, leaving a training
dataset of 27 601 training merger trees with 5 to 10 branches.

The data has shape (27 601, 3, 29, 10), where the first channel is the
distance variable, the second channel is the mass variable and the thirs
channel is the subhalo channel. To use the data, simply copy this code
after cloning the repo:

\begin{verbatim}
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = torch.load("path_to_repository_folder/data/dataset_normalized_consistent.pt", map_location = device)
\end{verbatim}

where you replace \textbf{path\_to\_repository\_folder} with the path
leading to the cloned repository.

\hypertarget{diffusion}{%
\subsubsection{diffusion}\label{diffusion}}

The diffusion folder contains two subfolders:
\textbf{Cornelius\_diffusion} and \textbf{diffusion\_test}.
\#\#\#\#\#diffusion\_test The diffusion\_test folder is for the model
and generated merger tree you create and generate through the
reproduction possibility of the project. This is done through
\textbf{train\_main.py}. If you were to reproduce the results or create
a new model, the model will be saved in
\textbf{diffusion/diffusion\_test/model/} and the generated merger trees
will be saved in
\textbf{diffusion/diffusion\_test/generated\_merger\_trees/}.

\#\#\#\#\#Cornelius\_diffusion The Cornelius\_diffusion folder contains
the existing model and a consistent generated merger trees dataset that
is created by me in the thesis. The Cornelius\_diffusion folder contains
a model subfolder where the model is stored and a subfolder called
generated\_merger\_trees where the generated merger trees are stored.
The model and generated merger trees are already pre-loaded to the
reproduction and analysis scripts in the folder with the correct paths.

you can load the model like this:

\begin{verbatim}
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("path_to_repository_folder/diffusion/Cornelius_diffusion/model/diffusion_model.pt", map_location = device)
\end{verbatim}

where you replace \textbf{path\_to\_repository\_folder} with the path
leading to the cloned repository. now \textbf{model} will contain the
state dict to the U-Net model. To generate a new sample, analyzing and
visualizing it, use the following code:

\begin{verbatim}
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
\end{verbatim}

To load the generated merger tree dataset use the following code:

\begin{verbatim}
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generated_merger_trees = torch.load("path_to_repository_folder/diffusion/Cornelius_diffusion/generated_merger_trees/generated_merger_trees.pt", map_location = device)
\end{verbatim}

where you replace \textbf{path\_to\_repository\_folder} with the path
leading to the cloned repository.

\hypertarget{experiment-important}{%
\subsubsection{experiment -- IMPORTANT}\label{experiment-important}}

The experiment folder contains all the useful functions for analyzing,
plotting, training, generating merger trees, and the diffusion model.

Some of the key functions in the folder are the following: 1.
\textbf{full\_evaluation in evaluation.py:} takes a dataset of merger
trees or a single merger tree of the shape (number of trees, 3, number
of snapshots, number of branches) and completely evaluate it. If an
original dataset is given, the mass ks statistic will also be evaluated.
This function is a great contribution toward fully evaluating generated
merger trees 2. \textbf{plot\_side\_by\_side in plot.py:} randomly
sample a merger tree from a given merger tree dataset and plot it in
three different ways. For more details about the functions, see the
functions in the scripts.

\hypertarget{files}{%
\subsection{Files}\label{files}}

\hypertarget{main_notebook.ipynb}{%
\subsubsection{Main\_notebook.ipynb}\label{main_notebook.ipynb}}

The main notebook is the main tool for evaluating and visualizing the
generated merger trees in different ways. The script contains 6
different code snippets that can be easily run. The code snippets do the
following: 1. Randomly sample a generated merger tree from the
consistent generated merger tree dataset in
\textbf{diffusion/Cornelius\_diffusion/generated\_merger\_trees/} and
analyze and visualize it. 2. Does the same as (1), but given the number
of branches you want the merger tree to contain. Simply change the
\textbf{num\_branches} variable to the desired number of branches you
want the sampled merger tree to contain. 3. Does the same as (2) but
with complexity instead of branches. Changing the two variables
\textbf{higher\_than} and \textbf{complexity} will randomly draw a
sample from the generated merger tree dataset with the desired
complexity. \textbf{higher\_than} is a True/False variable where True
means you want a higher complexity than the given complexity and False
means you want a lower complexity, and \textbf{complexity} ranges from
around 30 to 140. Sampling a complex merger tree would look like this:

\begin{verbatim}
higher_than = True
complexity = 120
\end{verbatim}

Sampling a sparse merger tree would look like this:

\begin{verbatim}
higher_than = False
complexity = 40
\end{verbatim}

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\setcounter{enumi}{3}
\tightlist
\item
  Combines (2) and (3), meaning you can draw a generated merger tree
  depending on the desired number of branches and complexity. It would
  be interesting to try different combinations of extreme values meaning
  few branches - high complexity, few branches - low complexity, many
  branches - high complexity and many branches -- low complexity. There
  are three variables to change to the desired values. Sampling a
  generated merger tree with many branches and high complexity would
  look like this:
\end{enumerate}

\begin{verbatim}
num_branches = 10
higher_than = True
complexity = 135
\end{verbatim}

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\setcounter{enumi}{4}
\tightlist
\item
  Use the diffusion model in
  \textbf{diffusion/Cornelius\_diffusion/model/} to generate a
  completely new merger tree, visualize and analyze it.
\item
  This code snippet deviates from (1) -- (5) and analyze and evaluate
  the whole generated merger tree dataset stored in
  \textbf{diffusion/Cornelius\_diffusion/generated\_merger\_trees/}.
  Note that plotting/visualizing a merger tree can give an error. This
  will only affect one out of the three plots. The function creating
  that plot sometimes cannot handle the randomly sampled merger tree.
  The sampled merger tree will still be visualized, but only by the two
  first plots. If this happens, run the code snippet again and it should
  draw a new merger tree and visualize it in all three different ways.
\end{enumerate}

\hypertarget{analyze_main.py}{%
\subsubsection{analyze\_main.py}\label{analyze_main.py}}

\textbf{Analyze\_main.py} does almost the same things as
\textbf{main\_notebook.ipynb}, but has less options. So for analyzing
and visual representation, I would suggest using the notebook.
\textbf{Analyze\_main.py} has three options: 1. \textbf{generate:}
generate a completely new sample, visualize and evaluate it. 2.
\textbf{random:} sample a random merger tree from the generated merger
tree dataset, visualize and evaluate it. 3. \textbf{analyze:} analyze
the full generated merger tree

Running the script can simply be done by typing the following in the
command line from the cloned folder:

\begin{verbatim}
$ python3 analyze_main.py
\end{verbatim}

The default is to generate a completely new sample (1).

To a´modify the script to do the other experiments, do the following:

\begin{verbatim}
$ python3 analyze_main.py --generate False --random False --analyze False
\end{verbatim}

where you replace \textbf{False} with \textbf{True} for the experiment
you want to do.

\hypertarget{train_main.py---reproduction}{%
\subsubsection{train\_main.py -
reproduction}\label{train_main.py---reproduction}}

\textbf{train\_main.py} is the main script fo reproduction. The script
has three main options: 1. \textbf{train:} train a completely new
diffusion model. 2. \textbf{generate:} generate a completely new merger
tree dataset of 10 000 merger trees. 3. \textbf{analyze:} analyze and
evaluate the generated merger tree dataset. The default setting is that
all of the above options are done in cronological order. But they can be
completed separately. The only thing to note is that (2) requires a new
diffusion model to generate new merger trees, which indicates that (1)
must have been completed at least once before running (2) and (3)
requires generated merger trees to be analyzed, which implies that (2)
must be completed at least once before running (3). My advice is to run
the default settings first.

To run the script, do the following:

\begin{verbatim}
$ python3 train_main.py
\end{verbatim}

and modify which experiment you want to run by changing \textbf{False}
to \textbf{True} in the following code:

\begin{verbatim}
$ python3 train_main.py --train False --generate False --analyze False
\end{verbatim}


(1) applies the consistent merger tree dataset in \textbf{data} to train a new diffusion model with the following hyperparameters which can be altered:
\begin{enumerate}
\item T = 1000 --T
\item batch\_size = 16 --batch\_size
\item epochs = 200 --epochs
\item lr = 3e-4 --lr 
\end{enumerate}

I would suggest that the two main hyperparameters to change are T and epochs. The trained model will be stored in \textbf{diffusion/diffusion\_test/model/} as \textbf{``model\_epochs=X1\_T=X2.pt''}, where X1 is the number of epochs chosen and X2 is the T value.


(2) requires a trained diffusion model, and the generated merger trees
will be saved in \textbf{diffusion/diffusion\_test/generated\_merger\_trees/} as
\textbf{``generated\_merger\_trees/model\_epochs=X1\_T=X2.pt''}. To
generate new merger tree from an existing model, make sure the
hyperparameter settings are the same as the existing model. However, I
suggest to train, generate and analyze in one go, for simplicity.
