from .buffer import Buffer
from uncertaintylearning.features.density_estimator import FixedSmoothKernelDensityEstimator
from uncertaintylearning.features.variance_estimator import GPVarianceEstimator, ZeroVarianceEstimator
from uncertaintylearning.features.distance_estimator import DistanceEstimator
from uncertaintylearning.models.mcdropout import MCDropout
from uncertaintylearning.models.ensemble import Ensemble
from uncertaintylearning.utils.density_picker import CrossValidator
from uncertaintylearning.utils.networks import create_optimizer
from uncertaintylearning.utils import invsoftplus
from torch.utils.data import TensorDataset, random_split
import torch
from uncertaintylearning.models import DEUP
from copy import deepcopy
from uncertaintylearning.features.feature_generator import FeatureGenerator
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation.sampling import MaxPosteriorSampling
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from gpytorch.utils.errors import NotPSDError


def init_buffer(networks, X_init, Y_init, features, domain=None, epsilon=1e-5, loggify=False, repeats=2, beta=None):
    if epsilon is None:
        raise Exception('epsilon cannot be None when initializing the buffer')
    if beta is not None:
        assert not loggify, "eiither log or invsoftplus"
    buffer = Buffer(X_init.size(-1), features=features)
    ds = TensorDataset(X_init, Y_init)
    for _ in range(repeats):

        td, vd = random_split(ds, [len(ds) // 2, len(ds) - len(ds) // 2])
        for (tds, vds) in [(td, vd), (vd, td)]:
            f_ = deepcopy(networks['f_predictor'])
            o = create_optimizer(f_, 1e-4)

            for epoch in range(300):
                o.zero_grad()
                y_hat = f_(tds[:][0])
                f_loss = torch.nn.MSELoss()(y_hat, tds[:][1])
                f_loss.backward()
                o.step()

            fg = make_feature_generator(features, tds[:][0], tds[:][1], domain, epsilon)
            features_lis = fg(X_init)

            buffer.add_features(features_lis)
            targets = (f_(X_init) - Y_init).pow(2).detach()
            if loggify:
                targets = targets.log().clamp_min_(-20)
            elif beta is not None:
                targets = invsoftplus(targets, beta).clamp_min_(-20)
            buffer.add_targets(targets)

    return buffer, fg


def make_feature_generator(features, X, Y, domain, epsilon=None, loggify=False, uvs=True, variance_model=None):
    dim = X.size(-1)
    density_estimator = None
    distance_estimator = None
    variance_estimator = None
    training_set = X if epsilon is not None else None

    if 'd' in features:
        # cv = CrossValidator(X.numpy(), Y.numpy(), alpha_search_space=[.1, .3, .5, .7, .9])
        # cv.fit()

        # density_estimator = FixedSmoothKernelDensityEstimator(bandwidth=cv.best_params_['bandwidth'],
        #                                                       alpha=cv.best_params_['alpha'],
        #                                                       use_density_scaling=False, domain=torch.cat((domain, X)))
        density_estimator = FixedSmoothKernelDensityEstimator(bandwidth=.1, alpha=.3, use_log_density=True, use_density_scaling=True,
                                                domain=torch.cat((domain, X)))

        density_estimator.fit(X)

    if 'v' in features:
        if variance_model is not None:
            model = variance_model
        else:
            model = SingleTaskGP(X, Y)
        variance_estimator = GPVarianceEstimator(model, loggify=loggify, use_variance_scaling=uvs, domain=domain)
        try:
            variance_estimator.fit()
        except NotPSDError:
            print('NotPSDError, using ZeroVariance estimator')
            variance_estimator = ZeroVarianceEstimator()
        # TODO: clean this
        # network = create_network(dim, 1, 128, 'relu', False, 5, 0.3)
        # optimizer = create_optimizer(network, 1e-4)
        # mcdrop = MCDropout(X, Y, network, optimizer, batch_size=64)
        # variance_estimator = VarianceSource(mcdrop, 50, 500)
        # variance_estimator.fit()

    if 'D' in features:
        distance_estimator = DistanceEstimator()
        distance_estimator.fit(X)

    return FeatureGenerator(features, density_estimator, distance_estimator, variance_estimator, training_set, epsilon)


def get_candidate(model, acq, full_train_Y, q, bounds, dim):
    if acq == 'EI':
        if q == 1:
            EI = ExpectedImprovement(model, full_train_Y.max().item())
        else:
            EI = qExpectedImprovement(model, full_train_Y.max().item())

        bounds_t = torch.FloatTensor([[bounds[0]] * dim, [bounds[1]] * dim])
        candidate, acq_value = optimize_acqf(
            EI, bounds=bounds_t, q=q, num_restarts=15, raw_samples=5000,
        )

    elif acq == 'TS':
        sobol = SobolEngine(dim, scramble=True)
        n_candidates = min(5000, max(20000, 2000 * dim))
        pert = sobol.draw(n_candidates)
        X_cand = (bounds[1] - bounds[0]) * pert + bounds[0]
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        candidate = thompson_sampling(X_cand, num_samples=q)

    else:
        raise NotImplementedError('Only TS and EI are implemented')

    return candidate, EI if acq == 'EI' else None


def one_step_acquisition_gp(oracle, full_train_X, full_train_Y, acq, q,  bounds, dim, domain, domain_image,
                            state_dict=None, plot_stuff=False):
    model = SingleTaskGP(full_train_X, full_train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    if state_dict is not None:
        model.load_state_dict(state_dict)
    fit_gpytorch_model(mll)

    candidate, EI = get_candidate(model, acq, full_train_Y, q, bounds, dim)

    if acq == 'EI' and dim == 1 and plot_stuff:
        plot_util(oracle, model, EI, domain, domain_image, None, full_train_X,
                  full_train_Y, candidate)

    candidate_image = oracle(candidate)
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, candidate_image])

    state_dict = model.state_dict()
    return full_train_X, full_train_Y, model, candidate, candidate_image, state_dict


def one_step_acquisition_ensemble(oracle, full_train_X, full_train_Y, networks, optimizers, epochs,
                                  acq, q,  bounds, dim, domain, domain_image, plot_stuff=False):
    model = Ensemble(full_train_X, full_train_Y, networks, optimizers, batch_size=64)
    model.fit(epochs)

    candidate, EI = get_candidate(model, acq, full_train_Y, q, bounds, dim)

    if acq == 'EI' and dim == 1 and plot_stuff:
        plot_util(oracle, model, EI, domain, domain_image, None, full_train_X,
                  full_train_Y, candidate)


    candidate_image = oracle(candidate)
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, candidate_image])

    state_dict = model.state_dict()
    return full_train_X, full_train_Y, model, candidate, candidate_image, state_dict


def one_step_acquisition_mcdropout(oracle, full_train_X, full_train_Y, networks, optimizers, epochs,
                                   acq, q,  bounds, dim, domain, domain_image, plot_stuff=False):
    # networks and optimizers are just a network and optimier
    model = MCDropout(full_train_X, full_train_Y, networks, optimizers)
    model.fit(epochs)

    candidate, EI = get_candidate(model, acq, full_train_Y, q, bounds, dim)

    if acq == 'EI' and dim == 1 and plot_stuff:
        plot_util(oracle, model, EI, domain, domain_image, None, full_train_X,
                  full_train_Y, candidate)


    candidate_image = oracle(candidate)
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, candidate_image])

    state_dict = model.state_dict()
    return full_train_X, full_train_Y, model, candidate, candidate_image, state_dict


def one_step_acquisition(oracle, full_train_X, full_train_Y, features, buffer, networks, optimizers,
                         domain, epsilon, epochs, candidate, candidate_image, acq, q, use_log_unc, estimator,
                         step, bounds, dim, f_losses, e_losses, print_stuff=False, plot_stuff=False, domain_image=None,
                         beta=None):
    if beta is not None:
        assert not use_log_unc
    fg = make_feature_generator(features, full_train_X, full_train_Y, domain, epsilon)
    data = {'x': full_train_X, 'y': full_train_Y}
    model = DEUP(data, fg, networks, optimizers,
                 exp_pred_uncert=use_log_unc,
                 estimator=estimator,
                 beta=beta)

    f_losses.extend(model.fit(epochs))

    if candidate is not None:  # which means candidate is now part of full_train_X
        features_lis = fg(candidate)
        buffer.add_features(features_lis)

        targets = (model.f_predictor(candidate) - candidate_image).pow(2).detach()
        if use_log_unc:
            targets = targets.log().clamp_min_(-10)
        elif beta is not None:
            targets = invsoftplus(targets, beta).clamp_min_(-10)
        buffer.add_targets(targets)

    e_losses.extend(model.fit_uncertainty_estimator(buffer.features, buffer.targets, epochs))
    if print_stuff:
        print_util(step, fg, full_train_X, full_train_Y, model, domain, domain_image, features)

    candidate, EI = get_candidate(model, acq, full_train_Y, q, bounds, dim)

    if acq == 'EI' and dim == 1 and plot_stuff:
        plot_util(oracle, model, EI, domain, domain_image, features, full_train_X,
                  full_train_Y, candidate, f_losses, e_losses)

    elif plot_stuff:
        plt.plot(f_losses, label='f')
        plt.plot(e_losses, label='e')

        plt.legend()
        plt.show()

    features_lis = fg(candidate)
    buffer.add_features(features_lis)
    candidate_image = oracle(candidate)
    targets = (model.f_predictor(candidate) - candidate_image).pow(2).detach()
    if use_log_unc:
        targets = targets.log().clamp_min_(-10)
    elif beta is not None:
        targets = invsoftplus(targets, beta).clamp_min_(-10)
    buffer.add_targets(targets)
    if print_stuff:
        print('\tMin buffer target: ', buffer.targets.squeeze().min().item(),
              ', max: ', buffer.targets.squeeze().max().item())
    full_train_X = torch.cat([full_train_X, candidate])
    full_train_Y = torch.cat([full_train_Y, candidate_image])

    return full_train_X, full_train_Y, model, candidate, candidate_image, buffer, fg


def print_util(step, fg, full_train_X, full_train_Y, model, domain, domain_image, features):
    print(f'step {step}, {full_train_X.shape[0]} datapoints seen')
    print('\tMean train error: ', torch.mean((model.predict(full_train_X) - full_train_Y).squeeze() ** 2).item())
    print('\tMean domain error: ', torch.mean((model.predict(domain) - domain_image).squeeze() ** 2).item())
    print('\tMean predicted train unc: ', torch.mean((model._uncertainty(x=full_train_X))).item())
    print('\tMean predicted domain unc: ', torch.mean((model._uncertainty(x=domain))).item())
    print('\tMean train unc-error: ', torch.mean((model._uncertainty(x=full_train_X) -
                                                  (model.predict(full_train_X) - full_train_Y) ** 2)).item())
    print('\tMean domain unc-error: ', torch.mean((model._uncertainty(x=domain) -
                                                   (model.predict(domain) - domain_image) ** 2) ** 2).item())
    if 'd' in features:
        print('\tKDE bandwidth: ', fg.density_estimator.kde.bandwidth)

    print('\tcurrent max:', full_train_Y.max().item())


def plot_util(oracle, model, EI, domain, domain_image, features, full_train_X,
              full_train_Y, candidate, f_losses=None, e_losses=None):
    eis = EI(domain.unsqueeze(1)).detach()
    max_ei, argmax_ei = torch.max(eis, 0)
    xmax = domain[argmax_ei].item()
    max_ei = max_ei.item()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
    # Plot optimization objective with noise level
    ax1.plot(domain, domain_image, 'y--', lw=2, label='Noise-free objective')
    ax1.plot(full_train_X, full_train_Y, 'kx', mew=3, label='Used for training')
    ax1.plot(candidate, oracle(candidate), 'rx', mew=3, label='candidate')

    ax1.plot(domain, model(domain).mean.detach().squeeze(), label='mean pred')
    ax1.fill_between(domain.numpy().ravel(),
                     model(domain).mean.detach().numpy().ravel() - model(domain).stddev.detach().numpy().ravel(),
                     model(domain).mean.detach().numpy().ravel() + model(domain).stddev.detach().numpy().ravel(),
                     alpha=.4)

    if features is not None:
        if 'd' in features:
            ax1.plot(domain, model.feature_generator.density_estimator.score_samples(domain), 'r-', label='density')

        if 'v' in features:
            ax1.plot(domain, model.feature_generator.variance_estimator.score_samples(domain), label='variance')


    ax2.plot(domain, eis, 'r-', label='EI')
    ax2.plot(domain, [max_ei] * len(domain), 'b--')
    ax2.plot([xmax] * 100, torch.linspace(0, max_ei, 100), 'b--')
    ax1.legend()
    ax2.legend()
    if f_losses is not None and e_losses is not None:
        ax3.plot(f_losses, label='f')
        ax3.plot(e_losses, label='e')
        # ax3.set_ylim(0, 10)
        ax3.legend()
    plt.show()


def optimize(oracle, bounds, X_init, Y_init, model_type="",
             networks=None, optimizers=None, features='xd', plot_stuff=False, n_steps=5,
             epochs=15, acq='EI', q=1, domain=None, epsilon=None, print_each=1, domain_image=None, use_log_unc=False,
             estimator='nn', beta=None):
    full_train_X = X_init
    full_train_Y = Y_init
    dim = X_init.size(-1)

    if beta is not None:
        assert not use_log_unc

    if model_type != "":
        state_dict = None
        buffer = None
    else:
        buffer, fg = init_buffer(networks, X_init, Y_init, features, domain, epsilon if epsilon is not None else 1e-5,
                             loggify=use_log_unc, beta=beta)
        if print_each < 10:
            print('\tInitial Buffer targets: ', buffer.targets.squeeze())
    f_losses = []
    e_losses = []

    max_value_per_step = [full_train_Y.max().item()]

    candidate = None
    candidate_image = None

    for step in range(n_steps):
        if model_type == "gp":
            outs = one_step_acquisition_gp(oracle, full_train_X, full_train_Y, acq, q,  bounds, dim, domain,
                                           domain_image, state_dict, plot_stuff=plot_stuff)
            full_train_X, full_train_Y, model, candidate, candidate_image, state_dict = outs
        elif model_type == "mcdropout":
            outs = one_step_acquisition_mcdropout(oracle, full_train_X, full_train_Y, networks, optimizers, epochs,
                                  acq, q,  bounds, dim, domain, domain_image, plot_stuff=plot_stuff)
            full_train_X, full_train_Y, model, candidate, candidate_image, state_dict = outs
        elif model_type == "ensemble":
            outs = one_step_acquisition_ensemble(oracle, full_train_X, full_train_Y, networks, optimizers, epochs,
                                  acq, q,  bounds, dim, domain, domain_image, plot_stuff=plot_stuff)
            full_train_X, full_train_Y, model, candidate, candidate_image, state_dict = outs
        else:
            outs = one_step_acquisition(oracle, full_train_X, full_train_Y, features, buffer, networks, optimizers,
                                        domain, epsilon, epochs, candidate, candidate_image, acq, q, use_log_unc, estimator,
                                        step, bounds, dim, f_losses, e_losses,
                                        print_stuff=(step + 1) % print_each == 0,
                                        plot_stuff=plot_stuff,
                                        domain_image=domain_image, beta=beta)
            full_train_X, full_train_Y, model, candidate, candidate_image, buffer, fg = outs
        max_value_per_step.append(full_train_Y.max().item())

    return max_value_per_step, f_losses, e_losses, buffer, model
