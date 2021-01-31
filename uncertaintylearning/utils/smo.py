from .buffer import Buffer
from .density_estimator import FixedSmoothKernelDensityEstimator, VarianceSource, DistanceEstimator
from ..models.mcdropout import MCDropout
from .density_picker import CrossValidator
from .networks import create_optimizer, create_network
from torch.utils.data import TensorDataset, random_split
import torch
import numpy as np
from ..models import EpistemicPredictor
from copy import deepcopy
from .feature_generator import FeatureGenerator
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation.sampling import MaxPosteriorSampling
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt


def init_buffer(networks, X_init, Y_init, features, domain=None, epsilon=1e-5):
    if epsilon is None:
        raise Exception('epsilon cannot be None when initializing the buffer')
    buffer = Buffer(X_init.size(-1), features=features)
    f_ = deepcopy(networks['f_predictor'])
    o = create_optimizer(f_, 1e-4)
    ds = TensorDataset(X_init, Y_init)
    for _ in range(3):
        tds, vds = random_split(ds, [len(ds) // 2, len(ds) - len(ds) // 2])

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
        buffer.add_targets(targets)
    return buffer


def make_feature_generator(features, X, Y, domain, epsilon=None):
    dim = X.size(-1)
    density_estimator = None
    distance_estimator = None
    variance_estimator = None
    training_set = X if epsilon is not None else None

    if 'd' in features:
        cv = CrossValidator(X.numpy(), Y.numpy(), alpha_search_space=[.1, .3, .5, .7, .9])
        cv.fit()

        density_estimator = FixedSmoothKernelDensityEstimator(bandwidth=cv.best_params_['bandwidth'],
                                                              alpha=cv.best_params_['alpha'],
                                                              use_density_scaling=True, domain=domain)

        density_estimator.fit(X)

    if 'v' in features:
        # TODO: clean this
        network = create_network(dim, 1, 128, 'relu', False, 5, 0.3)
        optimizer = create_optimizer(network, 1e-4)
        mcdrop = MCDropout(X, Y, network, optimizer, batch_size=64)
        variance_source = VarianceSource(mcdrop, 50, 500)
        variance_source.fit()

    if 'D' in features:
        distance_estimator = DistanceEstimator()
        distance_estimator.fit(X)

    return FeatureGenerator(features, density_estimator, distance_estimator, variance_estimator, training_set, epsilon)


def optimize(oracle, bounds, X_init, Y_init, networks, optimizers, features='xd', plot=False, n_steps=5,
             epochs=15, acq='EI', q=1, domain=None, epsilon=None, print_each=1, domain_image=None):
    full_train_X = X_init
    full_train_Y = Y_init
    dim = X_init.size(-1)

    buffer = init_buffer(networks, X_init, Y_init, features, domain, epsilon if epsilon is not None else 1e-5)

    f_losses = []
    e_losses = []
    state_dict = None
    max_value_per_step = [full_train_Y.max().item()]

    candidate = None
    candidate_image = None

    for step in range(n_steps):
        fg = make_feature_generator(features, full_train_X, full_train_Y, domain, epsilon)
        model = EpistemicPredictor(full_train_X, full_train_Y, fg, networks, optimizers)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        for _ in range(epochs):
            losses = model.fit()
            f_losses.append(np.mean(losses))

        if candidate is not None:  # which means candidate is part of full_train_X
            features_lis = fg(candidate)
            buffer.add_features(features_lis)

            targets = (model.f_predictor(candidate) - candidate_image).pow(2).detach()
            buffer.add_targets(targets)

        for _ in range(epochs):
            losses = model.fit_uncertainty_estimator(features=buffer.features, targets=buffer.targets)
            e_losses.append(np.mean(losses))

        if step % print_each == 0:
            print(f'step {step}, {full_train_X.shape[0]} datapoints seen')

            if 'd' in features:
                print('\tKDE bandwidth: ', fg.density_estimator.kde.bandwidth)

            print('\tcurrent max:', full_train_Y.max().item())

        if acq == 'EI':
            if q == 1:
                EI = ExpectedImprovement(model, full_train_Y.max().item())
            else:
                EI = qExpectedImprovement(model, full_train_Y.max().item())

            bounds_t = torch.FloatTensor([[bounds[0]] * dim, [bounds[1]] * dim])
            candidate, acq_value = optimize_acqf(
                EI, bounds=bounds_t, q=q, num_restarts=15, raw_samples=5000,
            )
            # print(candidate)

        elif acq == 'TS':
            sobol = SobolEngine(dim, scramble=True)
            n_candidates = min(5000, max(20000, 2000 * dim))
            pert = sobol.draw(n_candidates)
            X_cand = (bounds[1] - bounds[0]) * pert + bounds[0]
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            candidate = thompson_sampling(X_cand, num_samples=q)

        if acq == 'EI' and dim == 1 and plot:
            eis = EI(domain.unsqueeze(1)).detach()
            max_ei, argmax_ei = torch.max(eis, 0)
            xmax = domain[argmax_ei].item()
            max_ei = max_ei.item()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))
            # Plot optimization objective with noise level
            ax1.plot(domain, domain_image, 'y--', lw=2, label='Noise-free objective')
            # ax1.plot(X, f(X), 'rx', lw=1, alpha=0.2, label='Noisy samples')
            ax1.plot(full_train_X, full_train_Y, 'kx', mew=3, label='Used for training')
            ax1.plot(candidate, oracle(candidate), 'rx', mew=3, label='candidate')

            ax1.plot(domain, model(domain).mean.detach().squeeze(), label='mean pred')
            # print(model(domain).stddev.detach().numpy().ravel().mean())
            ax1.fill_between(domain.numpy().ravel(),
                             model(domain).mean.detach().numpy().ravel() - model(domain).stddev.detach().numpy().ravel(),
                             model(domain).mean.detach().numpy().ravel() + model(domain).stddev.detach().numpy().ravel(),
                             alpha=.4)
            if 'd' in features:
                ax1.plot(domain, model.feature_generator.density_estimator.score_samples(domain), 'r-', label='density')

            if 'v' in features:
                ax1.plot(domain, model.feature_generator.variance_estimator.score_samples(domain), label='variance')

            # ax1.set_ylim(-3, 2)

            ax2.plot(domain, eis, 'r-', label='EI')
            ax2.plot(domain, [max_ei] * len(domain), 'b--')
            ax2.plot([xmax] * 100, torch.linspace(0, max_ei, 100), 'b--')
            ax1.legend()
            ax2.legend()

            ax3.plot(f_losses, label='f')
            ax3.plot(e_losses, label='e')
            # ax3.set_ylim(0, 10)
            ax3.legend()
            plt.show()

        features_lis = fg(candidate)
        buffer.add_features(features_lis)
        candidate_image = oracle(candidate)
        targets = (model.f_predictor(candidate) - candidate_image).pow(2).detach()
        buffer.add_targets(targets)

        print('\tBuffer targets: ', buffer.targets.squeeze())

        full_train_X = torch.cat([full_train_X, candidate])
        full_train_Y = torch.cat([full_train_Y, candidate_image])

        max_value_per_step.append(full_train_Y.max().item())

    return max_value_per_step, f_losses, e_losses, buffer
