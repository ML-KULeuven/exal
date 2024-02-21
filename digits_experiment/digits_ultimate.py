import os
import time
import pickle
import wandb
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from sum_sampler import SumSampler
from data_generation import create_loader


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

BATCH_SIZE = 10 #10
EPOCHS = 20
DIGITS = 2
SAMPLES = 600 #100
LOSS = ''
ALPHA = 1. # determines diversity impact, 0. is no diversity
ANNEALED = True
NB_RUNS = 10

train_data, val_data, test_data = create_loader(DIGITS, BATCH_SIZE)

def train(seed):
    wandb.init(
        # set the wandb project where this run will be logged
        project="exal-digits",
        name=f"addition_{DIGITS}_annealed_{seed}",
        # track hyperparameters and run metadata
        config={
        "samples": SAMPLES,
        "experiment": "EXAL-digits",
        "epochs": EPOCHS,
        "digits": DIGITS,
        "batch_size": BATCH_SIZE,
        "loss": 'elbo_' + LOSS ,
        "alpha": ALPHA,
        "run": seed,
        "annealed": ANNEALED
        }
    )

    sum_sampler = SumSampler(DIGITS, SAMPLES, LOSS, ALPHA, BATCH_SIZE, annealed=ANNEALED)
    sum_sampler.train(train_data, val_data, test_data, EPOCHS)

    os.makedirs(f"digits_experiment/results/digits_{DIGITS}/", exist_ok=True)
    pickle.dump(sum_sampler.logger, open(f"digits_experiment/results/digits_{DIGITS}/logger_{seed}_samples{SAMPLES}_epochs{EPOCHS}_elbo{LOSS}_alpha{ALPHA}_logsumexp.pkl", "wb"))


if __name__ == "__main__":
    for i in range(NB_RUNS):
        train(i)