# Script for grid searching the best values for EP (comparing the results to GP)

import numpy as np
import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cmd-prefix', type=str, default='',
                    help='prefix to add to each run, can be slurm specific e.g.')
args = parser.parse_args()

n_steps = 100
# First GP with 10 seeds
functions = [
    'multi_optima',
    'levi_n13'
]
seeds = list(range(1, 11))
noises = [0, 0.1]

search_space = itertools.product(functions, seeds, noises)

for function, seed, noise in search_space:
    cl = f"""python optimize.py --gp --n-steps {n_steps} \\
             --seed {seed} --function {function} --noise {noise}
    """

    cl = args.cmd_prefix + cl
    print("Execute:")
    print("--------")
    print(cl)
    try:
        return_val = subprocess.check_output(cl, shell=True)
    except Exception:
        pass
    exit

# Now EP with varying values for epochs/retrain/batch_size/iid-ratio/bandwidth/kernel/e-lr
retrains = ["--retrain", ""]
epochss = [15, 50]
batch_sizes = [64, 16]
iid_ratios = [2 / 3, 5, 10]
# bandwidths = [0.05, 0.1]
# kernels = ['gaussian', 'exponential']
# elrs = [1e-3, 1e-4]

search_space = itertools.product(functions, seeds, noises, retrains, epochss, batch_sizes, iid_ratios)

i = 0

for function, seed, noise, retrain, epochs, batch_size, iid_ratio in search_space:

    cl = f"""python optimize.py --n-steps {n_steps} \\
             --seed {seed} --function {function} --noise {noise} \\
             --epochs {epochs} --batch_size {batch_size} --iid-ratio {iid_ratio} {retrain}
    """

    cl = args.cmd_prefix + cl
    print("Execute:")
    print("--------")
    print(cl)
    try:
        return_val = subprocess.check_output(cl, shell=True)
    except Exception:
        pass
    return_val = subprocess.check_output(cl, shell=True)
    exit
