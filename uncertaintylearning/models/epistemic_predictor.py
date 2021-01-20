import torch
import torch.nn as nn
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.quasirandom import SobolEngine
from uncertaintylearning.utils import DistanceEstimator

class EpistemicPredictor(Model):
    def __init__(self, train_X, train_Y,
                 networks,  # dict with keys 'a_predictor', 'e_predictor' and 'f_predictor'
                 optimizers,  # dict with keys 'a_optimizer', 'e_optimizer' and 'f_optimizer'
                 density_estimator=None,  # Instance of the DensityEstimator (..utils.density_estimator) class
                 train_Y_2=None,
                 schedulers=None,  # dict with keys 'a_scheduler', 'e_scheduler', 'f_scheduler', if empty, no scheduler!
                 split_seed=0,  # seed to randomly split iid from ood data
                 a_frequency=1,
                 batch_size=16,
                 iid_ratio=2/3,
                 dataloader_seed=1,
                 device=torch.device("cpu"),
                 retrain=True,
                 bounds=(-1, 2),
                 features='xd',  # x for input, d for density, D for distance, v for variance
                 variance_source=None,
                 ood_X=None,
                 ood_Y=None,
                 augmented_density=False,
                 ):
        super().__init__()

        if schedulers is None:
            schedulers = {}

        self.device = device
        self.features = features

        self.bounds = bounds

        if train_Y_2 is None:
            self.train_Y_2 = train_Y
        else:
            self.train_Y_2 = train_Y_2

        generator = torch.Generator().manual_seed(split_seed)

        dataset = TensorDataset(train_X, train_Y, self.train_Y_2)

        self.fake_data = iid_ratio if iid_ratio >= 1 else 0
        if ood_X is None and ood_Y is None:
            if self.fake_data > 0:
                train = dataset
                ood = self.generate_fake_data(dataset)
            else:
                n_train = int(iid_ratio * len(dataset))
                train, ood = random_split(dataset, (n_train, len(dataset) - n_train), generator=generator)
        else:
            train = dataset
            ood = TensorDataset(ood_X, ood_Y, ood_Y)
            self.fake_data = 0
        self.train_X, self.train_Y, self.train_Y_2 = train[:]
        self.ood_X, self.ood_Y, _ = ood[:]

        self.is_fitted = False

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.a_frequency = a_frequency
        self.actual_batch_size = min(batch_size, len(self.train_X) // 2)
        assert self.actual_batch_size >= 1, "Need more input points initially !"

        self.density_estimator = density_estimator
        self.variance_source = variance_source

        if 'D' in self.features:
            self.distance_estimator = DistanceEstimator()
            self.distance_estimator.fit(self.train_X)

        self.epoch = 0
        self.loss_fn = nn.MSELoss()

        self.a_predictor = networks['a_predictor']
        self.e_predictor = networks['e_predictor']
        self.f_predictor = networks['f_predictor']

        required_n_features = len(features) + (self.input_dim - 1 if 'x' in features else 0)
        assert self.e_predictor.input_layer.in_features == required_n_features, "Features don't match e network inputs"

        self.a_optimizer = optimizers['a_optimizer']
        self.e_optimizer = optimizers['e_optimizer']
        self.f_optimizer = optimizers['f_optimizer']

        self.schedulers = schedulers
        if schedulers is None:
            self.schedulers = {}

        self.data_so_far = (torch.empty((0, self.input_dim)).to(device),
                            torch.empty((0, self.output_dim)).to(device))

        self.ood_so_far = (torch.empty((0, self.input_dim)).to(device),
                           torch.empty((0, self.output_dim)).to(device))

        self.dataloader_seed = dataloader_seed
        self.augmented_density = augmented_density

        self.retrain = retrain

    def generate_fake_data(self, dataset):
        x, y, _ = dataset[:]
        length = int(self.fake_data * x.size(0))
        # ood_x = torch.FloatTensor(length, x.size(1)).uniform_(*self.bounds).to(self.device)

        sobol = SobolEngine(x.size(-1), scramble=True)
        pert = sobol.draw(length)
        X_cand = (self.bounds[1] - self.bounds[0]) * pert + self.bounds[0]
        Y_cand = torch.FloatTensor(length, y.size(1)).uniform_(y.min().item(), y.max().item()).to(self.device)
        #
        # X_cand = torch.empty((0, x.size(-1)))
        # Y_cand = torch.empty((0, y.size(1)))
        #
        # import torch.distributions as D
        # # mix = D.Categorical(torch.ones(x.size(0), ))
        # # comp = D.Independent(D.Normal(x, .5 * torch.ones_like(x)), 1)
        # # gmm = D.MixtureSameFamily(mix, comp)
        # # ood_x = gmm.sample(torch.Size([length]))
        # # ood_y = torch.FloatTensor(length, y.size(1)).uniform_(y.min().item(), y.max().item()).to(self.device)
        # # self.fake_data = int(self.fake_data)
        # cands = []
        # vals = []
        # for x_, y_ in zip(x, y):
        #     cs = D.Normal(x_, .1 * torch.ones_like(x_)).sample(torch.Size([self.fake_data]))
        #     X_cand = torch.cat([X_cand, cs])
        #     for c in cs:
        #         Y_cand = torch.cat([Y_cand, D.Normal(y_, .1 * torch.ones_like(y_)).sample().unsqueeze(1)])

        #ood_x = torch.cat(cands)
        #ood_y = torch.cat(vals).reshape(self.fake_data * y.size(0), y.size(1))
        ood_x = X_cand
        ood_y = Y_cand
        return TensorDataset(ood_x, ood_y, ood_y)

    @property
    def num_outputs(self):
        return self.output_dim

    def pretrain_density_estimator(self, x):
        """
        Trains density estimator on input samples
        """
        def perturb(data):
            import torch.distributions as D
            mix = D.Categorical(torch.ones(data.size(0), ))
            comp = D.Independent(D.Normal(data, .3 * torch.ones_like(data)), 1)
            gmm = D.MixtureSameFamily(mix, comp)
            xx = gmm.sample(torch.Size([20 * data.size(0)]))
            return xx

        if self.augmented_density:
            self.density_estimator.fit(torch.cat((x, perturb(x))).cpu())
        else:
            self.density_estimator.fit(x.cpu())

    def fit(self):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        train_losses = {'a': [], 'f': [], 'e': []}
        # TODO: maybe pretrain after duplicating dat a?
        data = TensorDataset(self.train_X, self.train_Y, self.train_Y_2)
        self.pretrain_density_estimator(self.train_X)
        if self.fake_data > 0:
            data = TensorDataset(*data[list(range(len(data))) * int(self.fake_data)])

        torch.manual_seed(self.dataloader_seed)
        loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        # ood_batch_size = max(1, self.actual_batch_size // (len(self.train_X) // len(self.ood_X)))
        ood_batch_size = max(1, (len(self.ood_X) * self.actual_batch_size) // (len(self.train_X)))

        ood_loader = DataLoader(TensorDataset(self.ood_X, self.ood_Y), shuffle=True, batch_size=ood_batch_size)
        ood_batches = list(ood_loader)

        for batch_id, (xi, yi, yi_2) in enumerate(loader):
            # Every `a_frequency` steps update a_predictor
            if self.epoch % self.a_frequency == 0:
                self.a_optimizer.zero_grad()
                a_hat = self.a_predictor(xi)
                a_loss = self.loss_fn(a_hat, (yi - yi_2).pow(2) / 2)
                a_loss.backward()
                self.a_optimizer.step()
                train_losses['a'].append(a_loss.item())
                xi = torch.cat([xi, xi], dim=0)
                yi = torch.cat([yi, yi_2], dim=0)

            if batch_id < len(ood_batches):
                ood_xi, ood_yi = ood_batches[batch_id]
            else:
                ood_xi = torch.empty((0, self.input_dim)).to(self.device)
                ood_yi = torch.empty((0, self.output_dim)).to(self.device)

            # Compute f_loss on unseen data and update
            f_loss, e_loss = self.train_with_batch(xi, yi, ood_xi, ood_yi)

            train_losses['f'].append(f_loss.item())
            train_losses['e'].append(e_loss.item())

            self.data_so_far = (torch.cat((self.data_so_far[0], xi), dim=0),
                                torch.cat((self.data_so_far[1], yi), dim=0))
            self.ood_so_far = (torch.cat((self.ood_so_far[0], ood_xi), dim=0),
                               torch.cat((self.ood_so_far[1], ood_yi), dim=0))

            # retrain on all data seen so far to update e for current f
            if self.retrain:
                self.retrain_with_collected()
        self.epoch += 1
        for scheduler in self.schedulers.values():
            scheduler.step()

        self.is_fitted = True
        return train_losses

    def retrain_with_collected(self):
        curr_loader = DataLoader(TensorDataset(*self.data_so_far), shuffle=True, batch_size=self.actual_batch_size)
        # ood_batch_size = max(1, len(self.ood_so_far[0]) // (len(self.data_so_far[0]) // self.actual_batch_size))
        ood_batch_size = max(1, (len(self.ood_X) * self.actual_batch_size) // (len(self.train_X)))

        ood_loader = DataLoader(TensorDataset(*self.ood_so_far), shuffle=True, batch_size=ood_batch_size)
        seen_ood_batches = list(ood_loader)
        for i, (prev_x, prev_y) in enumerate(curr_loader):
            if i < len(seen_ood_batches):
                prev_ood_x, prev_ood_y = seen_ood_batches[i]
            else:
                prev_ood_x = torch.empty((0, self.input_dim)).to(self.device)
                prev_ood_y = torch.empty((0, self.output_dim)).to(self.device)

            _ = self.train_with_batch(prev_x, prev_y, prev_ood_x, prev_ood_y)

    def train_with_batch(self, xi, yi, ood_xi, ood_yi):
        self.e_optimizer.zero_grad()
        x_ = torch.cat([xi, ood_xi], dim=0)
        y_ = torch.cat([yi, ood_yi], dim=0)
        f_loss = (self.f_predictor(x_) - y_).pow(2)
        with torch.no_grad():
            aleatoric = self.a_predictor(x_)
        loss_hat = self._epistemic_uncertainty(x_) + aleatoric
        e_loss = self.loss_fn(loss_hat, f_loss)
        e_loss.backward()
        self.e_optimizer.step()

        self.f_optimizer.zero_grad()
        y_hat = self.f_predictor(xi)
        f_loss = self.loss_fn(y_hat, yi)
        f_loss.backward()
        self.f_optimizer.step()

        return f_loss, e_loss

    def _epistemic_uncertainty(self, x):
        """
        Computes uncertainty for input sample and
        returns epistemic uncertainty estimate.
        """
        u_in = []
        if 'x' in self.features:
            u_in.append(x)
        if 'd' in self.features:
            density_feature = self.density_estimator.score_samples(x.cpu()).to(self.device)
            u_in.append(density_feature)
        if 'D' in self.features:
            distance_feature = self.distance_estimator.score_samples(x.cpu()).to(self.device)
            u_in.append(distance_feature)
        if 'v' in self.features:
            variance_feature = self.variance_source.score_samples(x.cpu()).to(self.device)
            u_in.append(variance_feature)
        u_in = torch.cat(u_in, dim=1)
        return self.e_predictor(u_in)

    def get_prediction_with_uncertainty(self, x):
        if x.ndim == 3:
            preds = self.get_prediction_with_uncertainty(x.view(x.size(0) * x.size(1), x.size(2)))
            return preds[0].view(x.size(0), x.size(1), 1), preds[1].view(x.size(0), x.size(1), 1)
        if not self.is_fitted:
            raise Exception('Model not fitted')
        return self.f_predictor(x), self._epistemic_uncertainty(x)

    def predict(self, x, return_std=False):
        # x should be a n x d tensor
        if not self.is_fitted:
            raise Exception('Model not fitted')
        self.eval()
        if not return_std:
            return self.f_predictor(x).detach()
        else:
            mean, var = self.get_prediction_with_uncertainty(x)
            return mean.detach(), var.detach().sqrt()

    def posterior(self, x, output_indices=None, observation_noise=False):
        # this works with 1d output only
        # x should be a n x d tensor
        mvn = self.forward(x)
        return GPyTorchPosterior(mvn)

    def forward(self, x):
        # ONLY WORKS WITH 1d output !!!!!
        # When x is of shape n x d, the posterior should have mean of shape n, and covar of shape n x n (diagonal)
        # When x is of shape n x q x d, the posterior should have mean of shape n x 1, and covar of shape n x q x q ( n diagonals)

        means, variances = self.get_prediction_with_uncertainty(x)

        # Sometimes the predicted variances are too low, and MultivariateNormal doesn't accept their range

        # TODO: maybe the two cases can be merged into one with torch.diag_embed
        if means.ndim == 2:
            mvn = MultivariateNormal(means.squeeze(), torch.diag(variances.squeeze() + 1e-6))
        elif means.ndim == 3:
            assert means.size(-1) == variances.size(-1) == 1
            try:
                mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
            except RuntimeError:
                print('RuntimeError')
                print(torch.diag_embed(variances.squeeze(-1)) + 1e-6)
        else:
            raise NotImplementedError("Something is wrong, just cmd+f this error message and you can start debugging.")
        return mvn
