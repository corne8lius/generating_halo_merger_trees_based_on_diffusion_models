#!/bin/bash
#SBATCH --job-name=Thesis
#SBATCH --account=ec35
#SBATCH --time=124:00:00
#SBATCH --partition=accel_long
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=250G
#
# by default, request two cores NumPy may just know how to take
# advantage of them; for larger computations, maybe use between
# six and ten; at some point, we will look at how to run on gpus
#
#SBATCH --cpus-per-task=2

# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md


source ${HOME}/.bashrc

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

# the important bit: unload all current modules (just in case) and load only the necessary ones
module purge
module load JupyterLab/3.5.0-GCCcore-11.3.0
source /fp/projects01/ec232/venvs/in5310/bin/activate

# print information (optional)
echo "submission directory: ${SUBMITDIR}"

# by default, pass on any remaining command-line options
python3 new_diffusion.py



# --model ${MODEL} --trainset ${TRAIN} --devset ${TEST}

# source /fp/projects01/ec232/venvs/in5310/bin/activate


# amir
# confidence training
# tail -f slurm-431247.out
# cat slurm-359048.out

# /fp/projects01/ec232/venvs/in5310/lib/python3.10/site-packages/litdata/
# /fp/projects01/ec232/g05/g05-p3
# scp -r ec-corneb@fox.educloud.no:/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other/generated_image_epochs=400_BILINEAR_T=800_linear.pt Documents/master/testing/good_model
# scp -r Documents/master/notebooks/data/dataset_normalized_consistent_only.pt ec-corneb@fox.educloud.no:/fp/projects01/ec35/homes/ec-corneb/data

scp -r ec-corneb@fox.educloud.no:/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other ec-corneb@fox.educloud.no:/fp/projects01/ec35/homes/ec-corneb/diffusion/test_other2