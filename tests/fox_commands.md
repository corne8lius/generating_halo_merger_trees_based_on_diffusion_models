ssh ec-corneb@fox.educloud.no

2380Brumunddal

our repo:

    cd /fp/projects01/ec35/homes/ec-corneb/master_code


run job:

    sbatch run.slurm


see inside queue:

    squeue

    squeue --me


see output:

    # see whole output (best after the job is done):
    cat slurm-479238.out

    # consistantly print output:
    tail -f slurm-479238.out

cancel job:

    scancel 431942

# diffusion 2.0 = 449744 --> 170 epochs in 20.5 hours, 7 timer å generere 10 000 bilder
# autoencoder diffusion = 449769


# remember to add epoch in name

617606 = 400 epochs, T = 950, linear, bilinear interpolation, normalized data, consistent only training data, batch_size = 32

610231 = 400 epochs, T = 800, linear, bilinear interpolation, normalized data, consistent only training data, batch_size = 32