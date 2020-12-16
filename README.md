# Learning to predict uncertainty

Experiments on learning to predict uncertainty.

## Install

```
https://github.com/MJ10/UncertaintyLearning.git
cd UncertaintyLearning/
conda create --name uncertainty python=3.7
conda activate uncertainty
pip install -e .
```

## Prepare for logging

If you want the results of your runs to be logged in a directory in `UncertaintyLearning` called `results`, you can run:
```
cd UncertaintyLearning/
mkdir results
conda activate uncertainty
conda env config vars set EP_EXPS_ROOT=$PWD/results
```

This creates an environment variable (available only when your conda environment is activated). The logging script in `uncertaintylearning/utils/logging.py` uses that environment
variable as root directory of all results.

## Bayesian Optimization Experiments
The main script for Bayesian optimization is `bin/optimize.py`. The acquisition function used for now is Expected Improvement.

The script takes as main arguments the function to optimize (as defined in `uncertaintylearning/utils/test_functions.py`), 
the number of steps, the number of initial points, the value of the std of the gaussian isotropic noise added to the 
deterministic function. 

If the flag `--gp` is specified, then the script uses a Gaussian Process to fit the data and estimate
the uncertainty. If the flag is not specified, then the Epistemic Predictor (EP - as defined in `uncertaintylearning/models/epistemic_predictor.py`)
is used, with neural networks defined in the script, and customizable with other arguments, as defined in the script. 
One of the main arguments when using EP is `--iid-ratio`. This number defines how examples with high uncertainty are generated and
given to the epistemic predictor: if the value is in $[0, 1)$, then at each step, the current available data is split into 2 groups, and only one of them
is used to train the main predictor (the function regressor), and the other group provides examples of high uncertainty as the main predictor cannot fit 
its values; if the value is larger than $1$, then at each step, data randomly generated is used as high uncertainty examples, and the `--iid-ratio`
determines how much data is randomly generated.


### Example of usage:
```bash
python bin/optimize.py --function booth --n-steps 50 --epochs 64 --batch-size 16 --iid-ratio 5 --kernel gaussian 
--bandwidth 0.1 --n-hidden 128 --f-lr 1e-4 --e-lr 5e-4
```


### Grid searching the best configuration
The script `bin/gs_optimize.py` can be used to compare many configurations of EP and compare them to GP. The script launches 
many processes, or (e.g. slurm) jobs, each of which runs the `optimize.py` script once with adequate arguments and flags.

The only argument of this script is `--cmd-prefix` and its current purpose is to make the script slurm-compatible.

Example of usage:
```bash
python bin/gs_optimize.py --cmd-prefix "sbatch run.sh "
```
, assuming `run.sh` is a bash script containing Slurm arguments, and ending with something similar to `exec $@`.

### Test time
When a particular configuration of EP is picked, it can be compared to GP on many different functions (with different random seeds),
using the `bin/run_jobs.py` script. It differs from `bin/gs_optimize.py` in that it doesn't loop through a grid of hyperparameters.

## Logging
As mentioned above, logging is handled by `uncertaintylearning/utils/logging.py`. Specifically, a subfolder containing a hash of the arguments (`argparse`) is created
in the root directory of results; inside which 2 `pickle` files are created. One containing the values of the arguments/hyperparameters, and one containing the results.
In the case of Bayesian Optimization, the results file contains one python list only, of length equal to `--n-steps`, representing the maximal value seen so far.

The notebook `notebooks/analyzeresults.ipynb` provides an example of how to analyze many results at once using `pandas` and `matplotlib`. 
You need to install `conda / pip` install `pandas` to run it !