import os

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, qMaxValueEntropy
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator,
                                       create_network, create_optimizer, create_multiplicative_scheduler)
from uncertaintylearning.models import EpistemicPredictor

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument("--n-runs", type=int, default=3,
                    help="number of runs for EP-EI and GP-EI optimizers")
parser.add_argument("--n-steps", type=int, default=2000,
                    help="number of optimization steps per run (budget)")
parser.add_argument("--noise", type=float, default=0.1,
                    help="standard deviation of gaussian noise added to deterministic objective")

parser.add_argument("--use-exp-log-density", action="store_true", default=False,
                    help="If specified, densities, instead of log densities, are used as input to epistemic predictor")
parser.add_argument("--use-density-scaling", action="store_true", default=False,
                    help="If specified, log densities are scaled to [0,1] before being input to epistemic predictor")

parser.add_argument("--epochs", type=int, default=5,
                    help="Number of epochs of training of the Epistemic predictor model")

parser.add_argument("--n-ood", type=int, default=7,
                    help="Number of OOD points to use at the first iteration")

parser.add_argument("--plot", action="store_true", default=False,
                    help="If specified, then for each run, all optimization steps are shown !")
parser.add_argument("--save", action="store_true", default=False,
                    help="If specified, all optimization results will be stored!")

# Arguments specific to density estimation
parser.add_argument("--cv-kernel", action="store_true", default=False,
                    help="If specified, a grid search for the best kernel and kernel parameters to use is performed")

parser.add_argument("--kernel", default='linear',
                    help="kernel to use in KDE. Ignored if --cv-kernel is specified.")
parser.add_argument("--bandwidth", type=float, default=0.7,
                    help="bandwith of kernel in KDE. Ignored if --cv-kernel is specified")

# Arguments specific to the neural networks, optimizers, and schedulers
parser.add_argument("--n-hidden", type=int, default=64,
                    help="number of neurons in the hidden layer for each network")
parser.add_argument("--a-lr", type=float, default=1e-2,
                    help="learning rate for the a network")
parser.add_argument("--f-lr", type=float, default=1e-2,
                    help="learning rate for the f network")
parser.add_argument("--e-lr", type=float, default=1e-3,
                    help="learning rate for the e network")
parser.add_argument("--a-wd", type=float, default=1e-6,
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = parser.parse_args()


def f(x, noise=args.noise, xmin=-1, xmax=1, yrange=2000):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    val = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1)) + noise * torch.randn(x.size(0))
    return - val.reshape(-1,1) / yrange


def rescale(x, a, b, c, d):
    """ Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


bounds = (-1, 1)
n_dim = 10

X_init = (bounds[1] - bounds[0]) * torch.rand(50, n_dim) + bounds[0]
Y_init = f(X_init)

# Plot optimization objective with noise level
if args.plot:
    X = torch.zeros(n_dim, torch.arange(bounds[0], bounds[1], 0.01).size(0))
    for n in range(n_dim):
        X[n, :] = torch.arange(bounds[0], bounds[1], 0.01)

    X_mesh = np.array(np.meshgrid(X[0, :], X[1, :]))
    Y = f(torch.tensor(np.transpose(X_mesh.reshape(n_dim, -1))), 0)
    fig = plt.figure()
    plt.contour(np.array(X_mesh[0]), np.array(X_mesh[1]), np.array(Y.reshape(-1, X.size(1))), lw=2, label='Noise-free objective', cmap=cm.PuBu_r)
    cbar = plt.colorbar()
    plt.scatter(np.array(X_init[:, 0]), np.array(X_init[:, 1]), c = np.array(Y_init), label='initial samples')
    plt.legend()
    cbar = plt.colorbar()
    plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(np.array(X_mesh[0]), np.array(X_mesh[1]), np.array(Y.reshape(-1, X.size(1))), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.scatter(np.array(X_init[:, 0]), np.array(X_init[:, 1]), np.array(Y_init), marker='o',  label='Noisy samples')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.legend()
    # plt.show()


def optimize(model_name, acqf_name, plot=False, n_steps=5, **kwargs):
    train_X = X_init
    train_Y = Y_init
    acquired_X = []
    acquired_Y = []

    ood_X = (bounds[1] - bounds[0]) * torch.rand(args.n_ood, n_dim) + bounds[0]
    additional_data = {'ood_X': ood_X,
                       'ood_Y': f(ood_X),
                       'train_Y_2': f(train_X)}
    state_dict = None
    max_value_per_step = [train_Y.max().item()]
    for n in range(n_steps):
        if model_name == EpistemicPredictor:
            model = model_name(train_X, train_Y, additional_data, **kwargs)
            if state_dict is not None:
                model.load_state_dict(state_dict)
            for _ in range(args.epochs):
                model.fit()
        elif model_name == SingleTaskGP:
            model = model_name(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            if state_dict is not None:
                model.load_state_dict(state_dict)
            fit_gpytorch_model(mll)
        else:
            raise Exception('Not sure this would work !')

        if acqf_name == 'EI':
            acqf = ExpectedImprovement(model, train_Y.max().item())
        elif acqf_name == 'UCB':
            acqf = UpperConfidenceBound(model, beta=beta)
        elif acqf_name == 'MES':
            candidate_set = torch.rand(100, 1)
            candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
            acqf = qMaxValueEntropy(model, candidate_set)

        bounds_t = torch.FloatTensor([[bounds[0]]*n_dim, [bounds[1]]*n_dim])
        candidate, acq_value = optimize_acqf(
            acqf, bounds=bounds_t, q=1, num_restarts=5, raw_samples=50,
        )

        train_X = torch.cat([train_X, candidate])
        acquired_X.append(candidate)
        train_Y = torch.cat([train_Y, f(candidate)])
        acquired_Y.append(f(candidate))

        additional_data['train_Y_2'] = torch.cat([additional_data['train_Y_2'], f(candidate)])
        state_dict = model.state_dict()

        if plot and n % (n_steps // 10) == 0:
            X_acqf = torch.tensor(np.transpose(X_mesh.reshape(n_dim, -1))).unsqueeze(1)
            acqf_score = acqf(X_acqf).detach()
            max_acqf_score, argmax_acqf_score = torch.max(acqf_score, 0)
            xmax = X_acqf[argmax_acqf_score]
            max_ucb = max_acqf_score.item()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
            # Plot optimization objective + Acquired points
            im11 = ax1.contourf(np.array(X_mesh[0]), np.array(X_mesh[1]), np.array(Y.reshape(-1, X.size(1))), lw=2,
                        label='Noise-free objective', cmap=cm.PuBu_r)
            plt.colorbar(im11, ax=ax1)
            ax1.scatter(np.array(X_init[:, 0]), np.array(X_init[:, 1]), c=np.array(Y_init), marker='o', label='initial samples')
            im12 = ax1.scatter(np.array(train_X[:, 0]), np.array(train_X[:, 1]), c=np.array(train_Y), marker='^', label='Acquired samples')
            plt.colorbar(im12, ax=ax1)
            ax1.legend()
            ax1.title.set_text('Schwefel function')
            
            # PLot acquisition function and its max point
            im21 = ax2.contourf(np.array(X_mesh[0]), np.array(X_mesh[1]), np.array(acqf_score.reshape(-1, X.size(1))),
                        cmap=cm.PuBu_r, label=acqf_name)
            plt.colorbar(im21, ax=ax2)
            im22 = ax2.scatter(np.array(xmax[0][0]), np.array(xmax[0][1]), c=np.array(max_ucb), marker='*',
                        label='max_acqf')
            plt.colorbar(im22, ax=ax2)
            ax2.legend()
            ax2.title.set_text(acqf_name + '_num_steps=' + str(n))
            if acqf_name == 'UCB':
                ax2.title.set_text(acqf_name+'_beta='+str(beta)+'_num_steps='+str(n))

            plt.show()

        max_value_per_step.append(train_Y.max().item())
        print(train_Y.max().item())

    return max_value_per_step

acqf_functions = ['UCB'] #['EI', 'UCB', 'MES']
beta = 10

save_path = 'results/schwefel/'+str(n_dim)+'d'
gp_runs = [np.zeros((args.n_runs, args.n_steps + 1)) for _ in range(len(acqf_functions))]
ep_runs = [np.zeros((args.n_runs, args.n_steps + 1)) for _ in range(len(acqf_functions))]

use_log_density = not args.use_exp_log_density
use_density_scaling = args.use_density_scaling
if not os.path.exists(save_path):
    os.mkdir(save_path)
for acqf_idx, acqf_name in enumerate(acqf_functions):

    for i in range(args.n_runs):
        print(f"Run {i + 1}/{args.n_runs}")
        gp_runs[acqf_idx][i] = optimize(SingleTaskGP, acqf_name, plot=args.plot, n_steps=args.n_steps)

        if args.cv_kernel:
            density_estimator = CVKernelDensityEstimator(use_log_density, use_density_scaling)
        else:
            density_estimator = FixedKernelDensityEstimator(args.kernel, args.bandwidth, use_log_density, use_density_scaling)

        # networks = {'a_predictor': create_network(1, 1, args.n_hidden, 'tanh', True),
        #             'e_predictor': create_network(2, 1, args.n_hidden, 'relu', True),
        #             'f_predictor': create_network(1, 1, args.n_hidden, 'relu', False)
        #             }
        #
        # optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], args.a_lr,
        #                                               weight_decay=args.a_wd,
        #                                               output_weight_decay=args.a_owd),
        #               'e_optimizer': create_optimizer(networks['e_predictor'], args.e_lr,
        #                                               weight_decay=args.e_wd,
        #                                               output_weight_decay=args.e_owd),
        #               'f_optimizer': create_optimizer(networks['f_predictor'], args.f_lr,
        #                                               weight_decay=args.f_wd,
        #                                               output_weight_decay=args.f_owd)
        #               }
        #
        # schedulers = {'a_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
        #                                                              lr_schedule=args.a_schedule),
        #               'e_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
        #                                                              lr_schedule=args.e_schedule),
        #               'f_scheduler': create_multiplicative_scheduler(optimizers['e_optimizer'],
        #                                                              lr_schedule=args.f_schedule)
        #               }
        #
        # ep_runs[acqf_idx][i] = optimize(EpistemicPredictor, acqf_name, plot=args.plot,
        #                       n_steps=args.n_steps,
        #                       networks=networks,
        #                       optimizers=optimizers,
        #                       schedulers=schedulers,
        #                       a_frequency=1,
        #                       density_estimator=density_estimator)
    if acqf_name == 'EI':
        save_name = acqf_name
    elif acqf_name == 'UCB':
        save_name = acqf_name+'_beta='+str(beta)

    if args.save:
        np.save(save_path+'/gp_' + acqf_name, gp_runs[acqf_idx])
        np.save(save_path+'/ep_' + acqf_name, ep_runs[acqf_idx])



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
for acqf_idx, acqf_name in enumerate(acqf_functions):
    ax1.errorbar(range(1 + args.n_steps), gp_runs[acqf_idx].mean(0), gp_runs[acqf_idx].std(0), ls='-',  label='Average maximum value reached by GP with '+acqf_name)
    # ax2.errorbar(range(1 + args.n_steps), ep_runs[acqf_idx].mean(0), ep_runs[acqf_idx].std(0), ls='--',  label='Average maximum value reached by EP with '+acqf_name)
ax1.legend()
ax2.legend()

ax1.grid()
ax2.grid()

plt.show()
