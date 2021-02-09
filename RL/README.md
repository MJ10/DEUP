This repo contains RL experiments code for [DEUP: Direct Epistemic Uncertainty Prediction]().  
DQN implementation is based on [here](https://github.com/pluebcke/dqn_experiments) and all the experiments use [bsuite](https://github.com/deepmind/bsuite).

### Requirements:

```
pip install -r requirements.txt
```

### Run:

To reproduce the results, simply run:
```
python main.py --agent ['dqn', 'deup_dqn', 'boot_dqn', 'mcdrop_dqn']
```
This creates a folder per each agent. 
To plot the results, run `show_results.py` after adding the save_path to the `experiments` dictionary.
