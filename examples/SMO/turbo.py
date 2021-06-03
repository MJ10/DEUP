# Copied and adapted from BoTorch tutorial on Turbo
import os
import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine


from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from uncertaintylearning.models.deup_gp import DEUP_GP

from .smo import make_feature_generator

from .buffer import Buffer
from torch.utils.data import TensorDataset, random_split
from gpytorch.utils.errors import NotPSDError

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
SMOKE_TEST = os.environ.get("SMOKE_TEST")

dim = 10
low, up = -10, 15

def eval_objective(x, dim=10):
    """This is a helper function we use to unnormalize and evalaute a point"""

    fun = Ackley(dim=dim, negate=True).to(dtype=dtype, device=device)
    fun.bounds[0, :].fill_(low)
    fun.bounds[1, :].fill_(up)
    dim = fun.dim
    lb, ub = fun.bounds
    return fun(unnormalize(x, fun.bounds))


@dataclass
class TurboState:
    dim: int
    batch_size: int
    step: int = 0
    max_step: int = 200
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    state.step += 1
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    # if state.length < state.length_min or state.step > state.max_step:
    if state.step > state.max_step:
        state.restart_triggered = True
    return state


def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
    deup=False,
    turbo=True,
):
    dim = X.shape[-1]
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    if not deup:
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    else:
        weights = model.f_predictor.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    if not turbo:
        tr_lb = torch.zeros(dim)
        tr_ub = torch.ones(dim)

    if acqf == "ts":
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=dtype, device=device)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        if batch_size > 1:
            ei = qExpectedImprovement(model, Y.max(), maximize=True)
        else:
            ei = ExpectedImprovement(model, Y.max(), maximize=True)
        try:
            X_next, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
        except NotPSDError:
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(batch_size).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert
            X_next = pert
            print('Warning: NotPSDError, using {} purely random candidates for this step'.format(batch_size))

    return X_next

batch_size = 4
n_init = 20  # 2*dim, which corresponds to 5 batches of 4

X_turbo = get_initial_points(dim, n_init)
Y_turbo = torch.tensor(
    [eval_objective(x, dim) for x in X_turbo], dtype=dtype, device=device
).unsqueeze(-1)

state = TurboState(dim, batch_size=batch_size)

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4

domain = (up - low) * torch.rand(1000, dim) + low


def init_buffer(X_init, Y_init, features, domain=None, epsilon=1e-5,
                loggify=False, repeats=5):
    if epsilon is None:
        raise Exception('epsilon cannot be None when initializing the buffer')
    buffer = Buffer(X_init.size(-1), features=features)
    ds = TensorDataset(X_init, Y_init)
    for _ in range(repeats):
        tds, vds = random_split(ds, [int(0.8 * len(ds)), len(ds) - int(0.8 * len(ds))])

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = SingleTaskGP(tds[:][0], tds[:][1], likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        fg = make_feature_generator(features, tds[:][0], tds[:][1], domain, epsilon,
                                    loggify=True, uvs=False, variance_model=model)
        features_lis = fg(X_init)
        buffer.add_features(features_lis)
        targets = (model(X_init).mean.unsqueeze(-1) - Y_init).pow(2).detach()

        if loggify:
            targets = targets.log().clamp_min_(-20)

        buffer.add_targets(targets)

    return buffer, fg


def train_turbo(X_turbo, Y_turbo, state=state, deup=False,
                features='xv',
                use_log_unc=True, domain=domain,
                batch_size=batch_size, acqf='ei', turbo=True, silent=True,
                dim=10, print_every=None):
    f_losses = []
    e_losses = []
    best_values = []
    candidate = None
    candidate_image = None
    buffer = None
    if deup:
        buffer, fg = init_buffer(X_turbo, (Y_turbo - Y_turbo.mean()) / Y_turbo.std(), features, domain,
                                 epsilon=1e-5,
                                 loggify=use_log_unc)

    while not state.restart_triggered:  # Run until TuRBO converges
        if print_every is not None and state.step % print_every == 0:
            print('step', state.step, state.best_value)
        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        if not deup:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X_turbo, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

        else:
            data = {'x': X_turbo, 'y': train_Y}
            model = DEUP_GP(data, fg, no_deup=False, exp_pred_uncert=use_log_unc)
            model.fit()
            fg = make_feature_generator(features, X_turbo, train_Y, domain=domain, epsilon=None,
                                        loggify=True, uvs=False, variance_model=model.f_predictor)
            model.feature_generator = fg

            if candidate is not None:  # which means candidate is now part of full_train_X
                features_lis = fg(candidate)
                buffer.add_features(features_lis)
                try:
                    targets = (model.f_predictor(candidate).mean.unsqueeze(-1) - candidate_image).pow(2).detach()
                except NotPSDError:
                    targets = (candidate_image - candidate_image).detach()
                if use_log_unc:
                    targets = targets.log().clamp_min_(-20)
                buffer.add_targets(targets)

            # e_losses.extend(model.fit_uncertainty_estimator(buffer.features, buffer.targets, epochs))
            model.fit_uncertainty_estimator(buffer.features, buffer.targets)

        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            X=X_turbo,
            Y=train_Y,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf=acqf,
            deup=deup,
            turbo=turbo
        )
        Y_next = torch.tensor(
            [eval_objective(x, dim) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)
        # print('Y_next', Y_next)

        if deup:
            # append candidate to buffer
            candidate, candidate_image = X_next, Y_next
            features_lis = fg(candidate)
            buffer.add_features(features_lis)
            targets = (model.f_predictor(candidate).mean.unsqueeze(-1) - candidate_image).pow(2).detach()

            if use_log_unc:
                targets = targets.log().clamp_min_(-10)

            buffer.add_targets(targets)
            # print('\tMin buffer target: ', buffer.targets.squeeze().min().item(),
            #           ', max: ', buffer.targets.squeeze().max().item())

        # Update state
        state = update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # Print current status
        if not silent:
            print(
                f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
            )

        best_values.append(state.best_value)

    return best_values, f_losses, e_losses, buffer

