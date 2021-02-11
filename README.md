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
We first train the main predictor, variance source and density source on the entire dataset and the spilts for training. The procedure is described in Appendix D.1. 
```bash
python bin/ood_pretrain.py --save_base_path <path_to_save_models> --data_base_path <path_to_store/load_data>
```

Next, we train the uncertainty predictor, using features and targets computed from the models trained above.
```bash
python bin/ood_train_deup.py --save_base_path <path_to_saved_models> --data_base_path <path_to_store/load_data> --features <feature_string>
```


## Sequential Model Optimization
The files used for SMO are mainly `models/epistemic_predictor.py` and `utils/smo.py`. 
The notebook `notebooks/SMO.ipynb` provides examples of usage, and cells which end up in the results of the paper. The notebook `notebooks/paperfigures.ipynb` makes plots with these results.


## Reinforcement Learning
The code for Reinforcement Learning experiments is in the `RL` folder: `RL/main.py` launches instances of the different RL agents, and `RL/show_results.py` makes plots with the obtained results.