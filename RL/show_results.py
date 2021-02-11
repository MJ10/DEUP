from bsuite.logging import csv_load
from bsuite.experiments import summary_analysis

from bsuite.experiments.bandit_noise import analysis as bandit_noise_analysis
from bsuite.experiments.deep_sea import analysis as deep_sea_analysis
from bsuite.experiments.cartpole import analysis as cartpole_analysis
from bsuite.experiments.mountain_car import analysis as mounrain_car_analysis

experiments = { 'DQN': "runs/DQN/",
                'DQN_DEUP': "runs/DQN_EP",
                "Boot_DQN": "runs/DQN_Boot",
                'DQN_MCdrop': "runs/DQN_MCdrop/"
                }

DF, SWEEP_VARS = csv_load.load_bsuite(experiments)
BSUITE_SCORE = summary_analysis.bsuite_score(DF, SWEEP_VARS)

env_id = 'cartpole'
if env_id.split('_')[0] == 'cartpole':
    analysis = cartpole_analysis
elif env_id.split('_')[0] == 'mountain':
    analysis = mounrain_car_analysis


df = DF[DF.bsuite_env == env_id].copy()
p = summary_analysis.plot_single_experiment(BSUITE_SCORE, env_id, SWEEP_VARS)
learning = analysis.plot_learning(df, SWEEP_VARS)
learning_seeds = analysis.plot_seeds(df, SWEEP_VARS)

print('Done')