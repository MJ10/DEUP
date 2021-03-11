import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.utils import pool
import torch
import warnings
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

parser = ArgumentParser()

# Main arguments for optimization
parser.add_argument("--agent", default='deup_dqn',
                    help='name of the agent')
parser.add_argument("--env", default='CARTPOLE',
                    help='name of the env')
parser.add_argument("--save_path", default="runs/DEUP_DQN/" ,
                    help='Path were the results are saved.')
args = parser.parse_args()

save_path = args.save_path


drop_prob = 0
if args.agent == 'deup_dqn':
    from agent_DEUP import Agent
elif args.agent == 'dqn':
    from agent import Agent
elif args.agent == 'boot_dqn':
    from bsuite.baselines.tf.boot_dqn import default_agent as Agent
elif args.agent == 'mcdrop_dqn':
    from agent import Agent
    drop_prob = 0.1
else:
    print('This agent is not implemented!!')


def run(bsuite_id: str) -> str:
    """
    Runs a bsuite experiment and saves the results as csv files

    Args:
        bsuite_id: string, the id of the bsuite experiment to run

    Returns: none

    """
    print(bsuite_id)
    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=save_path,
        logging_mode='csv',
        overwrite=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Settings for the neural network
    qnet_settings = {'layers_sizes': [50], 'batch_size': 64, 'noisy_nets': False, 'distributional': False, 'vmin': 0,
                     'vmax': 1000, 'number_atoms': 51, 'drop_prob': drop_prob}

    # Settings for the specific agent
    settings = {'batch_size': qnet_settings["batch_size"], 'epsilon_start': 0.05, 'epsilon_decay': 0.5,
                'epsilon_min': 0.05, 'gamma': 0.99, 'buffer_size': 2 ** 16, 'lr': 1e-3, 'qnet_settings': qnet_settings,
                'start_optimization': 64, 'update_qnet_every': 2, 'update_target_every': 50, 'ddqn': False, 'n_steps': 4,
                'duelling_dqn': False, 'prioritized_buffer': False, 'alpha': 0.6, 'beta0': 0.4, 'beta_increment': 1e-6}

    if args.agent == 'boot_dqn':
        agent = Agent(obs_spec=env.observation_spec(),
                      action_spec=env.action_spec()
                      )
    else:
        agent = Agent(action_spec=env.action_spec(),
                      observation_spec=env.observation_spec(),
                      device=device,
                      settings=settings
                      )



    experiment.run(
        agent=agent,
        environment=env,
        num_episodes=env.bsuite_num_episodes,
        verbose=False)
    return bsuite_id


bsuite_sweep = getattr(sweep, 'CARTPOLE')
pool.map_mpi(run, bsuite_sweep, 3)

