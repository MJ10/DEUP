# Script for running multiple jobs / processes comparing GP to a particular configuration of GP on different functions

import subprocess
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cmd-prefix', type=str, default='',
                    help='prefix to add to each run, can be slurm specific e.g.')
parser.add_argument('--no-gp', action="store_true", default=False,
                    help="If specified, only EP jobs are run.")
args = parser.parse_args()

# Just comment out the functions you don't want to try
functions = [
    ('multi_optima', 100),
    # ('levi_n13', 100),
    # ('booth', 100),
    # ('sinusoid', 100),
    # ('ackley10', 500)
]

seeds = list(range(1, 5))
noises = [0]  # , 0.1]

qs = (1, 5)

search_space = list(itertools.product(functions, seeds, noises))

# First, GP
if not args.no_gp:
    for function, seed, noise in search_space:
        function, n_steps = function
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

# Now, EP
iid_ratio = 5
batch_size = 64
epochs = 50


for function, seed, noise in search_space:
    function, n_steps = function
    for q in qs:
        cl = f"""python optimize.py --n-steps {n_steps} \\
                 --seed {seed} --function {function} --noise {noise} \\
                 --epochs {epochs} --batch_size {batch_size} --iid-ratio {iid_ratio} --q {q}
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
