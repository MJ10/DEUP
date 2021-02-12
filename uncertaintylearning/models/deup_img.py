import torch
import torch.nn as nn
from botorch.models.model import Model
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.stats

def to_one_hot(y, c=10):
    label = torch.FloatTensor(y.size(0), c).zero_()
    label = label.scatter(1, y.view(-1, 1), 1)
    return label

class DEUPEstimationImage(Model):
    def __init__(self,
                 data,
                 networks,  # dict with keys 'e_predictor' and 'f_predictor'
                 optimizers,  # dict with keys 'e_optimizer' and 'f_optimizer'
                 schedulers=None,  # dict with keys 'e_scheduler', 'f_scheduler', if empty, no scheduler!
                 density_estimator=None,  # Instance of the DensityEstimator (..utils.density_estimator) class
                 variance_source=None,
                 device=torch.device("cpu"),
                 features='dvb',  # x for input, d for density, v for variance, b is for seen/unseen bit
                 augmented_density=False,
                 loss_fn=nn.MSELoss()
                 ):
        super().__init__()
        self.device = device
        self.features = features

        self.train_loader = data['train_loader']
        # self.estimate_aleatoric = False

        self.is_fitted = False

        self.density_estimator = density_estimator
        self.variance_source = variance_source

        self.epoch = 0
        self.loss_fn = loss_fn

        self.e_predictor = networks['e_predictor']
        self.f_predictor = networks['f_predictor']

        self.e_optimizer = optimizers['e_optimizer']
        self.f_optimizer = optimizers['f_optimizer']

        self.schedulers = schedulers
        if schedulers is None:
            self.schedulers = {}

    @property
    def num_outputs(self):
        return self.output_dim

    def fit(self, epochs=100, val_loader=None):
        """
        Fit main predictor f to the input data
        """

        self.train()
        train_losses = {'a': [], 'f': []}

        for i in tqdm(range(epochs)):
            running_loss = 0.0
            self.f_predictor.train()
            for batch_id, data in enumerate(self.train_loader):
                xi, yi = data
                f_loss = self.train_with_batch(xi, yi)
                running_loss += f_loss.mean()
                if batch_id % 25 == 0:
                    print("batch {}, loss: {}".format(batch_id, running_loss / (batch_id + 1)))
                train_losses['f'].append(f_loss.item())
            
            if val_loader is not None:
                self.f_predictor.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, y = inputs.to(self.device), to_one_hot(targets).to(self.device)
                        outputs = self.f_predictor(inputs)
                        loss = self.loss_fn(outputs, y).sum(1).mean(0)
                        targets= targets.to(self.device)
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                acc = 100.*correct/total
                print("Epoch: {}, val acc: {}, val loss {}".format(i, acc, test_loss / total))
            
            self.epoch += 1
            for name, scheduler in self.schedulers.items():
                if name == 'f_scheduler':
                    scheduler.step()

        self.is_fitted = True
        return train_losses

    def get_epistemic_predictor_data(self, ood_loader, classes):
        epistemic_x = []
        epistemic_y = []
        epistemic_data = []
        
        with torch.no_grad():
            for x, y in ood_loader:
                u_out = torch.clamp(torch.log10(self.loss_fn(self.f_predictor(x.to(self.device)), to_one_hot(y).to(self.device)).sum(1)), min=-5).view(-1, 1)
                u_in = []
                if 'x' in self.features:
                    epistemic_data.append(x)
                if 'd' in self.features:
                    density_feature = self.density_estimator.score_samples(x, self.device, no_preprocess=False).to(self.device)
                    u_in.append(density_feature)
                if 'v' in self.features:
                    variance_feature = self.variance_source.score_samples(data=x, no_preprocess=False).to(self.device).view(-1, 1)
                    u_in.append(variance_feature)
                if 'b' in self.features:
                    u_in.append(torch.logical_or(y == classes[0], y==classes[1]).view(-1, 1).float().to(self.device))
                u_in = torch.cat(u_in, dim=1)
                epistemic_x.append(u_in)
                epistemic_y.append(u_out)
            epi_x, epi_y = torch.cat(epistemic_x, dim=0), torch.cat(epistemic_y, dim=0)
            if 'x' in self.features:
                epistemic_data = torch.cat(epistemic_data, dim=0)
        return epi_x, epi_y, epistemic_data


    def fit_ood(self, epochs=100, loader=None, classes=None, epistemic_loader=None):
        train_losses = {'e': []}
        self.e_predictor.train()
        if epistemic_loader is None:
            self.epistemic_X, self.epistemic_Y, self.epistemic_data = self.get_epistemic_predictor_data(loader, classes)
            if 'x' not in self.features:
                self.epistemic_loader = DataLoader(TensorDataset(self.epistemic_X, self.epistemic_Y), shuffle=True, batch_size=128)
            else: 
                self.epistemic_loader = DataLoader(TensorDataset(self.epistemic_X, self.epistemic_Y, self.epistemic_data), shuffle=True, batch_size=128)
        else:
            self.epistemic_loader = epistemic_loader
        for epoch in tqdm(range(epochs)):
            running_loss = 0
            for batch_id, data in enumerate(self.epistemic_loader):
                if 'x' in self.features:
                    epi_x, epi_y, epi_data = data
                    e_loss = self.train_ood_batch(epi_x, epi_y, epi_data)
                else:
                    epi_x, epi_y = data
                    e_loss = self.train_ood_batch(epi_x, epi_y)
                running_loss += e_loss.mean()
                train_losses['e'].append(e_loss.item())
                if batch_id % 50 == 0:
                    print("Iteration {}, Loss: {}".format(batch_id, running_loss / (batch_id + 1)))
            
            for name, scheduler in self.schedulers.items():
                if name == 'e_scheduler':
                    scheduler.step()
        return train_losses


    def train_with_batch(self, xi, yi):
        xi, yi = xi.to(self.device), to_one_hot(yi).to(self.device)
        self.f_optimizer.zero_grad()
        y_hat = self.f_predictor(xi)
        f_loss = self.loss_fn(y_hat, yi).sum(1).mean(0)
        f_loss.backward()
        self.f_optimizer.step()

        return f_loss

    def train_ood_batch(self, epi_x, epi_y, epi_data=None):
        self.e_optimizer.zero_grad()
        
        if epi_data is not None:
            epi_data = epi_data.to(self.device)
            loss_hat = self.e_predictor(epi_data, epi_x)
        else:
            loss_hat = self.e_predictor(epi_x)
        e_loss = (loss_hat - epi_y).pow(2)

        e_loss = e_loss.mean()
        e_loss.backward()
        self.e_optimizer.step()
        return e_loss

    def epistemic_uncertainty(self, x, is_unseen=False):
        """
        Computes uncertainty for input sample and
        returns epistemic uncertainty estimate.
        """
        u_in = []
        if 'd' in self.features:
            density_feature = self.density_estimator.score_samples(x, self.device).to(self.device)
            u_in.append(density_feature)
        if 'v' in self.features:
            variance_feature = self.variance_source.score_samples(data=x, no_preprocess=True).to(self.device).view(-1, 1)
            u_in.append(variance_feature)
        if 'b' in self.features:
            if not is_unseen:
                u_in.append(torch.zeros(x.shape[0]).to(self.device).view(-1, 1))
            else:
                u_in.append(torch.ones(x.shape[0]).to(self.device).view(-1, 1))
        u_in = torch.cat(u_in, dim=1)
        if 'x' in self.features:
            return self.e_predictor(x.to(self.device), u_in)
        else:
            return self.e_predictor(u_in)

    def get_prediction_with_uncertainty(self, x, features=None):
        if x.ndim == 3:
            assert features is None, "x cannot be of 3 dimensions, when features is explicitly given"
            preds = self.get_prediction_with_uncertainty(x.view(x.size(0) * x.size(1), x.size(2)))
            return preds[0].view(x.size(0), x.size(1), 1), preds[1].view(x.size(0), x.size(1), 1)

        if not self.is_fitted:
            raise Exception('Model not fitted')

        return self.f_predictor(x), self._uncertainty(features, x if features is None else None)

    def posterior(self, x, output_indices=None, observation_noise=False, **kwargs):
        features = None
        if 'features' in kwargs:
            features = kwargs['features']
        mvn = self.forward(x, features)
        return GPyTorchPosterior(mvn)

    def forward(self, x, features=None):
        means, variances = self.get_prediction_with_uncertainty(x, features)
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
