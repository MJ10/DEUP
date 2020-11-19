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

## Experiments

### Example of optimizing a one-dimensional function

```bash
python bin/optimize_1_D_function.py --cv-kernel --plot
```
The function is defined inside the script. The script also provides some helpful methods to extend the initially defined
function and its domain of definition in a pseudo-periodic way.

This script compares the performances of Bayesian Optimization with the Epistemic Uncertainty Predictor to that of
a Gaussian Process, with Expected Improvement acquisition function. It can easily be modified to include other
acquisition functions if needed. 