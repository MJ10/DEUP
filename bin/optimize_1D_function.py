import warnings
from argparse import ArgumentParser
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from uncertaintylearning.models import EpistemicPredictor
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator,
                                       create_network, create_optimizer, create_multiplicative_scheduler)

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument("--n-runs", type=int, default=1,
                    help="number of runs for EP-EI and GP-EI optimizers")
parser.add_argument("--n-steps", type=int, default=25,
                    help="number of optimization steps per run (budget)")
parser.add_argument("--initial-points", type=int, default=6,
                    help="Number of initial points.")
parser.add_argument("--compare-to-gp", action="store_true", default=False,
                    help="If specified, GP-EI models are trained as well.")
parser.add_argument("--noise", type=float, default=0.1,
                    help="standard deviation of gaussian noise added to deterministic objective")

parser.add_argument("--use-exp-log-density", action="store_true", default=False,
                    help="If specified, densities, instead of log densities, are used as input to epistemic predictor")
parser.add_argument("--use-density-scaling", action="store_true", default=False,
                    help="If specified, log densities are scaled to [0,1] before being input to epistemic predictor")

parser.add_argument("--epochs", type=int, default=15,
                    help="Number of epochs of training of the Epistemic predictor model")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Number of epochs of training of the Epistemic predictor model")

# Seeds for reproducibility
parser.add_argument("--split-seed", type=int, default=1,
                    help="Seed for splitting data into iid and ood")
parser.add_argument("--dataloader-seed", type=int, default=2,
                    help="Seed for creating shuffled dataloader")

# Arguments specific to density estimation
parser.add_argument("--cv-kernel", action="store_true", default=False,
                    help="If specified, a grid search for the best kernel and kernel parameters to use is performed")

parser.add_argument("--kernel", default='exponential',
                    help="kernel to use in KDE. Ignored if --cv-kernel is specified.")
parser.add_argument("--bandwidth", type=float, default=0.05,
                    help="bandwith of kernel in KDE. Ignored if --cv-kernel is specified")

# Arguments specific to the neural networks, optimizers, and schedulers
parser.add_argument("--n-hidden", type=int, default=64,
                    help="number of neurons in the hidden layer for each network")
parser.add_argument("--a-lr", type=float, default=1e-3,
                    help="learning rate for the a network")
parser.add_argument("--f-lr", type=float, default=1e-3,
                    help="learning rate for the f network")
parser.add_argument("--e-lr", type=float, default=1e-4,
                    help="learning rate for the e network")
parser.add_argument("--a-wd", type=float, default=0,
                    help="general weight decay for the a network")
parser.add_argument("--f-wd", type=float, default=0,
                    help="general weight decay for the f network")
parser.add_argument("--e-wd", type=float, default=0,
                    help="general weight decay for the e network")
parser.add_argument("--a-owd", type=float,
                    help="output layer weight decay for the a network")
parser.add_argument("--f-owd", type=float,
                    help="output layer weight decay for the f network")
parser.add_argument("--e-owd", type=float,
                    help="output layer weight decay for the e network")
parser.add_argument("--a-schedule", type=float,
                    help="multiplicative schedule for the a network")
parser.add_argument("--f-schedule", type=float,
                    help="multiplicative schedule for the f network")
parser.add_argument("--e-schedule", type=float,
                    help="multiplicative schedule for the e network")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: After a few tests, it looks like using CUDA does NOT improve speed. This probably is due to the numerous
# TODO: copying done, and the low batch sizes. Maybe needs more investigation...
device = torch.device("cpu")

args = parser.parse_args()

# Pick one of the two functions
def sinusoid_fct(X, noise=args.noise):
    return (-torch.sin(5 * X ** 2) - X ** 4 + 0.3 * X ** 3 + 2 * X ** 2 + 4.1 * X +
            noise * torch.randn_like(X))


def multi_optima_fct(X, noise=args.noise):
    return torch.sin(X) * torch.cos(5 * X) * torch.cos(22 * X) + noise * torch.randn_like(X)

# TODO: ideally, we should have a file containing all test functions we use, and the function is selected with argparse
f = multi_optima_fct

bounds = (-1, 2)

X = torch.arange(bounds[0], bounds[1], 0.01).reshape(-1, 1)
X = X.to(device)
Y = f(X, 0)


def optimize(model_name, X_init, Y_init, n_steps=5, f=multi_optima_fct, **kwargs):
    full_train_X = X_init
    full_train_Y = Y_init
    full_train_Y_2 = f(full_train_X)  # Set to None if uninterested in aleatoric uncertainty

    f_losses = []
    e_losses = []
    a_losses = []

    state_dict = None
    max_value_per_step = [full_train_Y.max().item()]
    for _ in range(n_steps):
        if model_name == EpistemicPredictor:
            use_log_density = not args.use_exp_log_density
            use_density_scaling = args.use_density_scaling
            if args.cv_kernel:
                density_estimator = CVKernelDensityEstimator(use_log_density, use_density_scaling)
            else:
                density_estimator = FixedKernelDensityEstimator(args.kernel, args.bandwidth, use_log_density,
                                                                use_density_scaling)
            model = model_name(full_train_X, full_train_Y, train_Y_2=full_train_Y_2,
                               density_estimator=density_estimator, device=device, **kwargs)
            model = model.to(device)
            if state_dict is not None:
                model.load_state_dict(state_dict)
            for _ in range(args.epochs):
                losses = model.fit()
                f_losses.append(np.mean(losses['f']))
                a_losses.append(np.mean(losses['a']))
                e_losses.append(np.mean(losses['e']))
        elif model_name == SingleTaskGP:
            model = model_name(full_train_X, full_train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            if state_dict is not None:
                model.load_state_dict(state_dict)
            fit_gpytorch_model(mll)
        else:
            raise Exception('Not sure this would work !')

        EI = ExpectedImprovement(model, full_train_Y.max().item())
        bounds_t = torch.FloatTensor([[bounds[0]], [bounds[1]]]).to(device)
        candidate, acq_value = optimize_acqf(
            EI, bounds=bounds_t, q=1, num_restarts=5, raw_samples=50,
        )

        full_train_X = torch.cat([full_train_X, candidate])
        full_train_Y = torch.cat([full_train_Y, f(candidate)])
        if model_name == EpistemicPredictor and full_train_Y_2 is not None:
            full_train_Y_2 = torch.cat([full_train_Y_2, f(candidate)])

        # TODO: check the effect of loading state dict VS retraining from scratch
        # state_dict = model.state_dict()
        max_value_per_step.append(full_train_Y.max().item())

    return max_value_per_step


if args.compare_to_gp:
    gp_runs = np.zeros((args.n_runs, args.n_steps + 1))
ep_runs = np.zeros((args.n_runs, args.n_steps + 1))

for i in range(args.n_runs):
    print(f"Run {i + 1}/{args.n_runs}")
    torch.manual_seed(i)
    X_init = (bounds[1] - bounds[0]) * torch.rand(args.initial_points, 1) + bounds[0]
    X_init = X_init.to(device)
    Y_init = f(X_init)

    if args.compare_to_gp:
        gp_runs[i] = optimize(SingleTaskGP, X_init, Y_init, n_steps=args.n_steps)

    networks = {'a_predictor': create_network(1, 1, args.n_hidden, 'tanh', True),
                'e_predictor': create_network(2, 1, args.n_hidden, 'relu', True),
                'f_predictor': create_network(1, 1, args.n_hidden, 'relu', False)
                }

    optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], args.a_lr,
                                                  weight_decay=args.a_wd,
                                                  output_weight_decay=args.a_owd),
                  'e_optimizer': create_optimizer(networks['e_predictor'], args.e_lr,
                                                  weight_decay=args.e_wd,
                                                  output_weight_decay=args.e_owd),
                  'f_optimizer': create_optimizer(networks['f_predictor'], args.f_lr,
                                                  weight_decay=args.f_wd,
                                                  output_weight_decay=args.f_owd)
                  }

    schedulers = {'a_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
                                                                 lr_schedule=args.a_schedule),
                  'e_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
                                                                 lr_schedule=args.e_schedule),
                  'f_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
                                                                 lr_schedule=args.f_schedule)
                  }

    ep_runs[i] = optimize(EpistemicPredictor, X_init, Y_init,
                          n_steps=args.n_steps,
                          networks=networks,
                          optimizers=optimizers,
                          schedulers=schedulers,
                          a_frequency=1,
                          batch_size=args.batch_size,
                          split_seed=args.split_seed,
                          dataloader_seed=args.dataloader_seed)
    print("Current results of EP so far:", ep_runs[:(i + 1)])
    if args.compare_to_gp:
        print("Current results of GP so far:", gp_runs[:(i + 1)])

# TODO: add something to save the results into csv for example, also save args and all parameters

