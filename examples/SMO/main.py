import torch
from uncertaintylearning.utils import create_network, create_optimizer
import numpy as np

from test_functions import functions, bounds as boundsx
from smo import optimize
from uncertaintylearning.models.mcdropout import MCDropout
import pickle

from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--n-seeds", type=int, default=1, help='number of seeds')
parser.add_argument("--n-steps", type=int, default=50, help='number of optimization steps')
parser.add_argument("--n-init", type=int, default=6, help='number of initial datapoints')
parser.add_argument("--function", default='multi_optima', help='one of the keys of SMO.test_functions.functions')
parser.add_argument("--noise", type=float, default=0, help='additive aleatoric noise')
parser.add_argument("--method", default='deup', help='one of deup, gp, mcdropout, ensemble')
parser.add_argument("--save_base_path", default='.', help='path to save results')

args = parser.parse_args()

fct_name = args.function
fct = functions[fct_name]
dim, bounds = boundsx[fct_name]
noise = args.noise
f = lambda x: fct(x, args.noise)

X = (bounds[1] - bounds[0]) * torch.rand(1000, dim) + bounds[0]
Y = f(X)  # This is for visualization purposes, the model never has access to this !
if dim == 1:
    X = torch.arange(bounds[0], bounds[1], 0.01).reshape(-1, 1)
    Y = f(X)

n_seeds = args.n_seeds
n_steps = args.n_steps

results = np.zeros((n_seeds, 1 + n_steps))
use_log_unc = True
features = 'xv'

for seed in range(n_seeds):
    torch.manual_seed(10 + seed)
    X_init = (bounds[1] - bounds[0]) * torch.rand(args.n_init, dim) + bounds[0]
    Y_init = f(X_init)

    print(f'Seed {seed}, Y_init_max {Y_init.max().item()}')

    if args.method == 'gp':
        outs = optimize(f, bounds, X_init, Y_init, model_type="gp", plot_stuff=False, domain=X, domain_image=Y,
                        n_steps=n_steps)

    elif args.method == 'ensemble':
        nets = [create_network(dim, 1, 128, 'relu', False, 3) for i in range(3)]
        opts = [create_optimizer(nets[i], 1e-3) for i in range(3)]
        outs = optimize(f, bounds, X_init, Y_init, model_type="ensemble", networks=nets, optimizers=opts,
                        features=features,
                        epochs=200, plot_stuff=False, domain=X, domain_image=Y, n_steps=n_steps)

    elif args.method == 'mcdropout':
        network = create_network(dim, 1, 128, 'relu', False, 3, 0.3)
        optimizer = create_optimizer(network, 1e-3)
        mcdropout_model = MCDropout(X_init, Y_init, network, optimizer, batch_size=64)
        outs = optimize(f, bounds, X_init, Y_init, model_type="mcdropout", networks=network, optimizers=optimizer,
                        features=features,
                        epochs=200, plot_stuff=False, domain=X, domain_image=Y, n_steps=n_steps)

    elif args.method == 'deup':
        networks = {
            'e_predictor': create_network(len(features) + (dim - 1 if 'x' in features else 0),
                                          1, 128, 'relu', False if use_log_unc else True, 3),
            'f_predictor': create_network(dim, 1, 128, 'relu', False, 3)
        }
        optimizers = {
            'e_optimizer': create_optimizer(networks['e_predictor'], 1e-3),
            'f_optimizer': create_optimizer(networks['f_predictor'], 1e-3)
        }
        outs = optimize(f, bounds, X_init, Y_init, networks=networks, optimizers=optimizers, features=features,
                        plot_stuff=False,
                        n_steps=n_steps, epochs=200, domain=X, domain_image=Y, print_each=100, use_log_unc=True,
                        estimator='gp')
    results[seed] = outs[0]

string = f"{args.method}_{args.function}_{args.n_init}"
filename = os.path.join(args.save_base_path, string)
pickle.dump({'results': results, 'string': string}, open(filename, 'wb'))

print('Results saved !')
