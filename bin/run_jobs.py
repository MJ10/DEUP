# Script for running multiple jobs / processes comparing GP to a particular configuration of GP on different functions

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
    'levi_n13',
    'booth',
    'sinusoid'
]
seeds = list(range(1, 11))
noises = [0, 0.1]

search_space = list(itertools.product(functions, seeds, noises))

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

# Now EP
iid_ratio = 10
batch_size = 64
epochs = 50

# bandwidths, kernels, elrs)

for function, seed, noise, in search_space:
    cl = f"""python optimize.py --n-steps {n_steps} \\
             --seed {seed} --function {function} --noise {noise} \\
             --epochs {epochs} --batch_size {batch_size} --iid-ratio {iid_ratio} 
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
