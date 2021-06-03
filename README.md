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

## Uncertainty estimation with a fixed dataset
The notebook `notebooks/fixed_training_set.ipynb` illustrates how DEUP is used to train an epistemic uncertainty predictor


## Rejecting Difficult Examples
Install DUE:
```bash
pip install git+https://github.com/y0ast/DUE.git
```

We first train the main predictor, variance source and density source on the entire dataset and the spilts for training. The procedure is described in Appendix D.1. This script should take about a day to run on a V100 GPU.
```bash
python examples/ood_detection/ood_pretrain.py --save_base_path <path_to_save_models> --data_base_path <path_to_store/load_data>
```

Next, we train the uncertainty predictor, using features and targets computed from the models trained above.
```bash
python examples/ood_detection/ood_train_deup.py --load_base_path <path_to_saved_models> --data_base_path <path_to_store/load_data> --features <feature_string>
```


## Sequential Model Optimization
The notebook `notebooks/SMO.ipynb` provides examples of usage. The notebook `notebooks/paperfigures.ipynb` makes plots with the results obtained by the `SMO` notebook.

Alternatively, you can run
```bash
python examples/SMO/main.py
```
, with adequate arguments (`--method`, `--n-steps`, etc...) to optimize a function and save the results in a pickle file.

To compare Turbo-EI and Turbo-DEUP-EI, the notebooks   `notebooks/turbo-ackleyloop.ipynb` and `notebooks/turbodeup.ipynb` allow to recreate the figures in the paper/


## Reinforcement Learning
DQN implementation is based on [here](https://github.com/pluebcke/dqn_experiments) and all the experiments use [bsuite](https://github.com/deepmind/bsuite).

To reproduce the results of the paper, simply run:
```
python examples/RL/main.py --agent ['dqn', 'deup_dqn', 'boot_dqn', 'mcdrop_dqn']
```
This creates a folder per each agent. 
To plot the results, run `python/RL/show_results.py` after adding the save_path to the `experiments` dictionary.

To use `boot_dqn`, install [bsuite[baselines]](https://github.com/deepmind/bsuite/tree/master/bsuite/baselines) first.
```
pip install bsuite[baselines]
```
