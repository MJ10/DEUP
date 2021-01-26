import torch
import torch.nn as nn
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal

from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.quasirandom import SobolEngine
from uncertaintylearning.utils import DistanceEstimator

class DEUP(Model):
    def __init__(self, 
                 data,
                 networks,  # dict with keys 'a_predictor', 'e_predictor' and 'f_predictor'
                 optimizers,  # dict with keys 'a_optimizer', 'e_optimizer' and 'f_optimizer'
                 schedulers=None,  # dict with keys 'a_scheduler', 'e_scheduler', 'f_scheduler', if empty, no scheduler!
                 split_seed=0,  # seed to randomly split iid from ood data
                 batch_size=16,
                 density_estimator=None,  # Instance of the DensityEstimator (..utils.density_estimator) class
                 variance_source=None,
                 iid_ratio=2/3,
                 device=torch.device("cpu"),
                 use_dataloaders=True,
                 bounds=(-1, 2),
                 features='xd',  # x for input, d for density, D for distance, v for variance
                 augmented_density=False,
                 loss_fn=nn.MSELoss(),
                 a_frequency=1,
                 dataloader_seed=1,
                 retrain=False
                 ):
        super().__init__()

        if schedulers is None:
            schedulers = {}

        self.device = device
        self.features = features
        self.bounds = bounds
        self.use_dataloaders = use_dataloaders
        if self.use_dataloaders:
            self.train_loader = data['train_loader']
            self.ood_loader = data['ood_loader']
            # train_loader = data['train_loader']
            self.estimate_aleatoric = False
        else:
            train_X = data['train_X']
            train_Y = data['train_Y']

            generator = torch.Generator().manual_seed(split_seed)

            
            if 'train_Y_2' in data.keys():
                self.train_Y_2 = data['train_Y_2']
                self.estimate_aleatoric = True
            else:
                self.train_Y_2 = train_Y
                self.estimate_aleatoric = False

            dataset = TensorDataset(train_X, train_Y, self.train_Y_2)
            self.fake_data = iid_ratio if iid_ratio >= 1 else 0
    
            if 'ood_X' not in data.keys():
                if self.fake_data > 0:
                    train = dataset
                    ood = self.generate_fake_data(dataset)
                else:
                    n_train = int(iid_ratio * len(dataset))
                    train, ood = random_split(dataset, (n_train, len(dataset) - n_train), generator=generator)
            else:
                train = dataset
                ood_X = torch.cat([data['ood_X'], train_X], dim=0)
                ood_Y = torch.cat([data['ood_Y'], train_Y], dim=0)
                ood = TensorDataset(ood_X, ood_Y, ood_Y)
                self.fake_data = 0
        
            self.train_X, self.train_Y, self.train_Y_2 = train[:]
            self.ood_X, self.ood_Y, _ = ood[:]


        self.is_fitted = False

        self.input_dim = train_X.size(1)
        self.output_dim = train_Y.size(1)

        self.a_frequency = a_frequency
        # self.actual_batch_size = min(batch_size, len(self.train_X) // 2)
        # assert self.actual_batch_size >= 1, "Need more input points initially !"
        self.actual_batch_size = 32

        self.density_estimator = density_estimator
        self.variance_source = variance_source

        if 'D' in self.features:
            self.distance_estimator = DistanceEstimator()
            self.distance_estimator.fit(self.train_X)

        self.epoch = 0
        self.loss_fn = loss_fn

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

    def fit(self, epochs=100):
        """
        Update a,f,e predictors with acquired batch
        """

        self.train()
        train_losses = {'a': [], 'f': []}
        if self.estimate_aleatoric or not self.use_dataloaders:
            # TODO: maybe pretrain after duplicating data?
            data = TensorDataset(self.train_X, self.train_Y, self.train_Y_2)
            # if self.fake_data > 0:
            #     data = TensorDataset(*data[list(range(len(data))) * int(self.fake_data)])

            torch.manual_seed(self.dataloader_seed)
            self.train_loader = DataLoader(data, shuffle=True, batch_size=self.actual_batch_size)
        
        for i in range(epochs):
            for batch_id, data in enumerate(self.train_loader):
                if self.estimate_aleatoric:
                    xi, yi, yi_2 = data
                else:
                    xi, yi = data
                # Every `a_frequency` steps update a_predictor
                if self.epoch % self.a_frequency == 0 and self.estimate_aleatoric:
                    self.a_optimizer.zero_grad()
                    a_hat = self.a_predictor(xi)
                    a_loss = nn.MSELoss()(a_hat, (yi - yi_2).pow(2) / 2)
                    a_loss.backward()
                    self.a_optimizer.step()
                    train_losses['a'].append(a_loss.item())
                    xi = torch.cat([xi, xi], dim=0)
                    yi = torch.cat([yi, yi_2], dim=0)

                f_loss = self.train_with_batch(xi, yi)# , ood_xi, ood_yi)

                train_losses['f'].append(f_loss.item())

            self.epoch += 1
            if i % 50 == 0:
                print("Epoch {}".format(i))
            for name, scheduler in self.schedulers.items():
                if name == 'f_scheduler' or name == 'e_scheduler':
                    scheduler.step()

        if self.retrain:
            self.retrain_with_collected()

        self.is_fitted = True
        return train_losses
    
    def get_epistemic_predictor_data(self, ood_loader):
        epistemic_x = []
        epistemic_y = []
        with torch.no_grad():
            for x, y in ood_loader:
                u_out = (self.f_predictor(x) - y).pow(2)
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
                epistemic_x.append(u_in)
                epistemic_y.append(u_out)
            epi_x, epi_y = torch.cat(epistemic_x, dim=0), torch.cat(epistemic_y, dim=0)
        return epi_x, epi_y
    
    def fit_ood(self, epochs=100):
        assert self.is_fitted, "Please train F first"
        train_losses = {'e': []}
        
        if not self.use_dataloaders:
            ood_batch_size = max(1, (len(self.ood_X) * self.actual_batch_size) // (len(self.train_X)))
            self.ood_loader = DataLoader(TensorDataset(self.ood_X, self.ood_Y), shuffle=True, batch_size=ood_batch_size)

        self.epistemic_X, self.epistemic_Y = self.get_epistemic_predictor_data(self.ood_loader)
        self.epistemic_loader = DataLoader(TensorDataset(self.epistemic_X, self.epistemic_Y), shuffle=True, batch_size=32)
        
        for epoch in range(epochs):
            for batch_id, (epi_x, epi_y) in enumerate(self.epistemic_loader):
                e_loss = self.train_ood_batch(epi_x, epi_y)
                train_losses['e'].append(e_loss.item())
            if epoch % 50 == 0:
                print("Epoch {}".format(epoch))
            for name, scheduler in self.schedulers.items():
                if name == 'e_scheduler':
                    scheduler.step()
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

    def train_with_batch(self, xi, yi):
        self.f_optimizer.zero_grad()
        y_hat = self.f_predictor(xi)
        f_loss = self.loss_fn(y_hat, yi)
        f_loss.backward()
        self.f_optimizer.step()

        return f_loss

    def train_ood_batch(self, epi_x, epi_y):
        self.e_optimizer.zero_grad()
        # f_loss = (self.f_predictor(ood_x) - ood_y).pow(2)
        # with torch.no_grad():
        #     aleatoric = self.a_predictor(ood_x)
        loss_hat = self.e_predictor(epi_x) #+ aleatoric
        e_loss = self.loss_fn(loss_hat, epi_y)
        e_loss.backward()
        self.e_optimizer.step()
        return e_loss

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
