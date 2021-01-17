import warnings
from argparse import ArgumentParser
import numpy as np
import torch
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from uncertaintylearning.models import EpistemicPredictor, MCDropout, Ensemble
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator, functions, bounds,
                                       compute_exp_dir, log_args, log_results, create_network, create_optimizer,
                                       create_multiplicative_scheduler, reset_weights)

warnings.filterwarnings('ignore')

parser = ArgumentParser()

# Main arguments for optimization
parser.add_argument("--function", default='multi_optima',
                    help='name of the function to optimize')
parser.add_argument("--n-steps", type=int, default=25,
                    help="number of optimization steps per run (budget)")
parser.add_argument("--initial-points", type=int, default=6,
                    help="Number of initial points.")
parser.add_argument("--gp", action="store_true", default=False,
                    help="If specified, this will run a GP-EI model")
parser.add_argument("--mcdrop", action="store_true", default=False,
                    help="If specified, this will run a MCDropout-EI model")
parser.add_argument("--ensemble", action="store_true", default=False,
                    help="If specified, this will run a Ensemble-EI model")
parser.add_argument("--seed", type=int, default=0,
                    help="seed for initial data generation, and NN initialization")
parser.add_argument("--noise", type=float, default=0.,
                    help="standard deviation of gaussian noise added to deterministic objective")
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="If specified, CPU is used even if CUDA is available")

# Arguments specific to EP
parser.add_argument("--epochs", type=int, default=15,
                    help="Number of epochs of training of the Epistemic predictor model")
parser.add_argument("--batch_size", type=int, default=64,
                    help="Number of epochs of training of the Epistemic predictor model")
parser.add_argument("--a-frequency", type=int, default=1,
                    help="Frequency of updating aleatoric estimator")
parser.add_argument("--iid-ratio", type=float, default=2/3,
                    help="ratio of iid data. If larger than one, all iid data is used, and fake data is used as OOD")
parser.add_argument("--retrain", action="store_true", default=False,
                    help="If specified, then retraining is done at the end of each minibatch pass")
parser.add_argument("--reset-weights", action="store_true", default=False,
                    help="If specified, then weights are reinitialized at every step")

# Arguments specific to density estimation for EP
parser.add_argument("--cv-kernel", action="store_true", default=False,
                    help="If specified, a grid search for the best kernel and kernel parameters to use is performed")
parser.add_argument("--kernel", default='exponential',
                    help="kernel to use in KDE. Ignored if --cv-kernel is specified.")
parser.add_argument("--bandwidth", type=float, default=0.05,
                    help="bandwidth of kernel in KDE. Ignored if --cv-kernel is specified")
parser.add_argument("--use-exp-log-density", action="store_true", default=False,
                    help="If specified, densities, instead of log densities, are used as input to epistemic predictor")
parser.add_argument("--use-density-scaling", action="store_true", default=False,
                    help="If specified, log densities are scaled to [0,1] before being input to epistemic predictor")

# Seeds for reproducibility for EP
parser.add_argument("--split-seed", type=int, default=1,
                    help="Seed for splitting data into iid and ood")
parser.add_argument("--dataloader-seed", type=int, default=2,
                    help="Seed for creating shuffled dataloader")

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

parser.add_argument("--dropout_prob", type=float,
                    help="dropout rate for mcdropout network")
parser.add_argument("--lengthscale", type=float,
                    help="lengthscale for mcdropout")
parser.add_argument("--tau", type=float,
                    help="tau for mcdropout")

parser.add_argument("--num_members", type=int,
                    help="number of ensemble members")


args = parser.parse_args()

# Log arguments in a new directory
exp_dir = compute_exp_dir(args)
log_args(args, exp_dir)

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
# TODO: After a few tests, it looks like using CUDA does NOT improve speed. This probably is due to the numerous
# TODO: copying done, and the low batch sizes. Maybe needs more investigation...

function = functions[args.function]

dim, bounds = bounds[args.function]  # Bounds of the corresponding hypercube

if args.seed != 0:
    torch.manual_seed(args.seed)
X_init = (bounds[1] - bounds[0]) * torch.rand(args.initial_points, dim) + bounds[0]
Y_init = function(X_init, args.noise).to(device)
if args.noise != 0:
    Y_init_2 = function(X_init, args.noise).to(device)
else:
    Y_init_2 = None
X_init = X_init.to(device)

if not args.gp and not args.mcdrop:
    networks = {'a_predictor': create_network(dim, 1, args.n_hidden, 'tanh', True),
                'e_predictor': create_network(dim + 1, 1, args.n_hidden, 'relu', True),
                'f_predictor': create_network(dim, 1, args.n_hidden, 'relu', False)
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

    f_losses = []
    e_losses = []
    a_losses = []

if args.mcdrop:
    lengthscale = args.lengthscale
    tau = args.tau
    dropout_prob = args.dropout_prob
    # density_estimator = FixedKernelDensityEstimator('exponential', 0.05)
    #torch.manual_seed(8)
    networks = {
        'f_predictor': create_network(dim, 1, args.n_hidden, 'relu', False, dropout_prob)
    }

if args.ensemble:
    networks = [create_network(dim, 1, args.n_hidden, 'relu', False) for _ in range(args.num_members)]
    optimizers = [create_optimizer(networks[i], args.f_lr,
                                                  weight_decay=args.f_wd,
                                                  output_weight_decay=args.f_owd) for i in range(args.num_members)]
full_train_X = X_init
full_train_Y = Y_init
full_train_Y_2 = Y_init_2

max_value_per_step = [full_train_Y.max().item()]
state_dict = None
for step in range(args.n_steps):
    if args.gp:
        model = SingleTaskGP(full_train_X, full_train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        fit_gpytorch_model(mll)
    elif args.mcdrop:
        reg = lengthscale ** 2 * (1 - dropout_prob) / (2. * full_train_X.size(0) * tau)
        optimizers['f_optimizer'] = create_optimizer(networks['f_predictor'], args.f_lr,
                                                                            weight_decay=reg,
                                                                            output_weight_decay=None)
        model = MCDropout(full_train_X, full_train_Y, network=networks['f_predictor'], optimizer=optimizers['f_optimizer'], batch_size=args.batch_size)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        for _ in range(args.epochs):
            model.fit()
    elif args.ensemble:
        model = Ensemble(full_train_X, full_train_Y, networks=networks, optimizers=optimizers, batch_size=args.batch_size, device=device)
        model = model.to(device)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        for _ in range(args.epochs):
            model.fit()
    else:
        if args.cv_kernel:
            density_estimator = CVKernelDensityEstimator(not args.use_exp_log_density, args.use_density_scaling)
        else:
            density_estimator = FixedKernelDensityEstimator(args.kernel, args.bandwidth, not args.use_exp_log_density,
                                                            args.use_density_scaling)
        model = EpistemicPredictor(train_X=full_train_X,
                                   train_Y=full_train_Y,
                                   networks=networks,
                                   optimizers=optimizers,
                                   density_estimator=density_estimator,
                                   train_Y_2=full_train_Y_2,
                                   schedulers=schedulers,
                                   split_seed=args.split_seed,
                                   a_frequency=args.a_frequency,
                                   batch_size=args.batch_size,
                                   iid_ratio=args.iid_ratio,
                                   dataloader_seed=args.dataloader_seed,
                                   device=device,
                                   retrain=args.retrain,
                                   bounds=bounds)
        model = model.to(device)
        if state_dict is not None and not args.reset_weights:
            model.load_state_dict(state_dict)
        if args.reset_weights:
            reset_weights(model)
        for _ in range(args.epochs):
            losses = model.fit()
            f_losses.append(np.mean(losses['f']))
            a_losses.append(np.mean(losses['a']))
            e_losses.append(np.mean(losses['e']))

    EI = ExpectedImprovement(model, full_train_Y.max().item())
    bounds_t = torch.FloatTensor([[bounds[0]] * dim, [bounds[1]] * dim]).to(device)
    candidate, acq_value = optimize_acqf(
        EI, bounds=bounds_t, q=1, num_restarts=5, raw_samples=50,
    )
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, function(candidate.cpu(), args.noise).to(device)])
    if full_train_Y_2 is not None:
        full_train_Y_2 = torch.cat([full_train_Y_2, function(candidate.cpu(), args.noise).to(device)])

    # TODO: check the effect of loading state dict VS retraining from scratch
    state_dict = model.state_dict()
    max_value_per_step.append(full_train_Y.max().item())
    print(max_value_per_step)

    # Log results
    log_results(max_value_per_step, exp_dir)

