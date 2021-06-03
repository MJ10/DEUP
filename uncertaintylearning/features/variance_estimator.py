import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import SoftmaxLikelihood

from due.dkl import DKL_GP, GP, initial_values_for_GP
from uncertaintylearning.utils.resnet import ResNet18Spec

# Taken from https://github.com/y0ast/deterministic-uncertainty-quantification/blob/master/utils/resnet_duq.py
class ResNet_DUQ(nn.Module):
    def __init__(
            self,
            input_size,
            num_classes,
            centroid_size,
            model_output_size,
            length_scale,
            gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.resnet = models.resnet18(pretrained=False, num_classes=model_output_size)

        # Adapted resnet from:
        # https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = nn.Identity()

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.resnet(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.resnet(x)
        y_pred = self.rbf(z)

        return z, y_pred


def bce_loss_fn(y_pred, y, num_classes):
    bce = F.binary_cross_entropy(y_pred, y, reduction="sum").div(
        num_classes * y_pred.shape[0]
    )
    return bce


def calc_gradients_input(x, y_pred):
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]

    gradients = gradients.flatten(start_dim=1)

    return gradients


def calc_gradient_penalty(x, y_pred):
    gradients = calc_gradients_input(x, y_pred)

    # L2 norm
    grad_norm = gradients.norm(2, dim=1)

    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()

    return gradient_penalty


class DUQVarianceSource:
    def __init__(self, input_size, num_classes, centroid_size, model_output_size, length_scale, gamma,
                 l_gradient_penalty, device):
        self.l_gradient_penalty = l_gradient_penalty
        self.device = device
        self.num_classes = num_classes
        self.model = ResNet_DUQ(input_size, num_classes, centroid_size, model_output_size, length_scale, gamma)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[25, 50, 75], gamma=0.2
        )
        self.postprocessor = MinMaxScaler()

    def fit(self, epochs=75, train_loader=None, save_path=None, val_loader=None):
        self.model.train()
        for epoch in tqdm(range(epochs)):
            running_loss = 0
            for i, (x, y) in enumerate(train_loader):
                self.model.train()

                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                if self.l_gradient_penalty > 0:
                    x.requires_grad_(True)

                z, y_pred = self.model(x)
                y = F.one_hot(y, self.num_classes).float()

                loss = bce_loss_fn(y_pred, y, self.num_classes)

                if self.l_gradient_penalty > 0:
                    loss += self.l_gradient_penalty * calc_gradient_penalty(x, y_pred)
                running_loss += loss.mean()
                loss.backward()
                self.optimizer.step()

                x.requires_grad_(False)

                with torch.no_grad():
                    self.model.eval()
                    self.model.update_embeddings(x, y)

                if i % 50 == 0:
                    print("Iteration: {}, Loss = {}".format(i, running_loss / (i + 1)))

            if epoch % 1 == 0 and val_loader is not None:
                self.model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, y = inputs.to(self.device), F.one_hot(targets, self.num_classes).float().to(self.device)
                        z, outputs = self.model(inputs)
                        loss = bce_loss_fn(outputs, y, self.num_classes)
                        targets = targets.to(self.device)
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.to(self.device)).sum().item()
                acc = 100. * correct / total
                print("Epoch: {}, test acc: {}, test loss {}".format(epoch, acc, test_loss / total))

            self.scheduler.step()

        if save_path is not None:
            torch.save(self.model, save_path)

    def score_samples(self, data=None, loader=None, no_preprocess=False):
        self.model.eval()

        with torch.no_grad():
            scores = []

            if loader is None:
                data = data.to(self.device)
                output = self.model(data)[1]
                kernel_distance, pred = output.max(1)

                scores.append(-kernel_distance.cpu())

            else:
                for data, target in loader:
                    data = data.to(self.device)
                    # target = target.cuda()

                    output = self.model(data)[1]
                    kernel_distance, pred = output.max(1)

                    scores.append(-kernel_distance.cpu())

        scores = torch.cat(scores, dim=0)
        if no_preprocess:
            values = scores.numpy().ravel()
        else:
            values = self.postprocessor.transform(scores.unsqueeze(-1)).squeeze()
        values = torch.FloatTensor(values).unsqueeze(-1)
        return values


class DUEVarianceSource:
    def __init__(self, input_size, num_classes, spectral_normalization, n_power_iterations, batchnorm_momentum,
                n_inducing_points, learning_rate, weight_decay, ard, kernel, separate_inducing_points, lengthscale_prior, coeff, device):
        self.device = device
        self.num_classes = num_classes
        self.n_inducing_points = n_inducing_points
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.ard = ard
        self.kernel = kernel
        self.separate_inducing_points = separate_inducing_points
        self.lengthscale_prior = lengthscale_prior
        self.coeff = coeff

        self.feature_extractor = ResNet18Spec(input_size, spectral_normalization, n_power_iterations, batchnorm_momentum)
        
        
        self.postprocessor = MinMaxScaler()

    def save(self, path):
        torch.save(self.model.state_dict(), path + "model.pt")
        torch.save(self.likelihood.state_dict(), path + "likelihood.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "model.pt"))
        self.likelihood.load_state_dict(torch.load(path + "likelihood.pt"))
        self.model.to(self.device)
        self.likelihood = self.likelihood.to(self.device)

    def fit(self, epochs=75, train_loader=None, save_path=None, val_loader=None):
        initial_inducing_points, initial_lengthscale = initial_values_for_GP(
            train_loader.dataset, self.feature_extractor, self.n_inducing_points
        )

        self.gp = GP(
            num_outputs=self.num_classes,
            initial_lengthscale=initial_lengthscale,
            initial_inducing_points=initial_inducing_points,
            separate_inducing_points=self.separate_inducing_points,
            kernel=self.kernel,
            ard=self.ard,
            lengthscale_prior=self.lengthscale_prior,
        )

        self.model = DKL_GP(self.feature_extractor, self.gp)
        self.model.to(self.device)

        self.likelihood = SoftmaxLikelihood(num_classes=10, mixing_weights=False)
        self.likelihood = self.likelihood.to(self.device)

        self.elbo_fn = VariationalELBO(self.likelihood, self.gp, num_data=len(train_loader.dataset))

        parameters = [
            {"params": self.feature_extractor.parameters(), "lr": self.learning_rate},
            {"params": self.gp.parameters(), "lr": self.learning_rate},
            {"params": self.likelihood.parameters(), "lr": self.learning_rate},
        ]

        self.optimizer = torch.optim.SGD(
            parameters, momentum=0.9, weight_decay=self.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[25, 50, 75], gamma=0.2
        )

        self.model.train()
        for epoch in tqdm(range(epochs)):
            running_loss = 0
            for i, (x, y) in enumerate(train_loader):
                self.model.train()

                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                y_pred = self.model(x)
                elbo = -self.elbo_fn(y_pred, y)
                running_loss += elbo.item()
                elbo.backward()
                self.optimizer.step()
                
                if i % 50 == 0:
                    print("Iteration: {}, Loss = {}".format(i, running_loss / (i + 1)))

            if epoch % 1 == 0 and val_loader is not None:
                self.model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, y = inputs.to(self.device), F.one_hot(targets, self.num_classes).float().to(self.device)
                        y_pred = self.model(data).to_data_independent_dist()
                        output = self.likelihood(y_pred).probs.mean(0)
                        predicted = torch.argmax(output, dim=1)
                        loss = -self.likelihood.expected_log_prob(y, y_pred).mean()
                        test_loss += loss.item()
                        targets = targets.to(self.device)
                        total += targets.size(0)
                        correct += predicted.eq(targets.to(self.device)).sum().item()
                acc = 100. * correct / total
                print("Epoch: {}, test acc: {}, test loss {}".format(epoch, acc, test_loss / total))

            self.scheduler.step()

        if save_path is not None:
            self.save(save_path)

    def score_samples(self, data=None, loader=None, no_preprocess=False):
        self.model.eval()

        with torch.no_grad():
            scores = []

            if loader is None:
                data = data.to(self.device)
                y_pred = self.model(data).to_data_independent_dist()
                output = self.likelihood(y_pred).probs.mean(0)

                scores.append(-(output * output.log()).sum(1).cpu())

            else:
                for data, target in loader:
                    data = data.to(self.device)
                    # target = target.cuda()
                    y_pred = self.model(data).to_data_independent_dist()
                    output = self.likelihood(y_pred).probs.mean(0)
                    
                    output = self.likelihood(y_pred).probs.mean(0)

                    scores.append(-(output * output.log()).sum(1).cpu())

        scores = torch.cat(scores, dim=0)
        if no_preprocess:
            values = scores.numpy().ravel()
        else:
            values = self.postprocessor.transform(scores.unsqueeze(-1)).squeeze()
        values = torch.FloatTensor(values).unsqueeze(-1)
        return values


class ZeroVarianceEstimator:
    def score_samples(self, test_points):
        return torch.zeros((test_points.size(0), 1))


class GPVarianceEstimator:
    def __init__(self, gp_model, loggify=False, use_variance_scaling=False, domain=None, fitted=False):
        self.gp_model = gp_model
        self.fitted = fitted
        self.loggify = loggify
        self.postprocessor = None
        if use_variance_scaling:
            self.postprocessor = MinMaxScaler()
        self.domain = None if not use_variance_scaling else domain

    def fit(self):
        if not self.fitted:
            mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
            fit_gpytorch_model(mll)
        if self.domain is not None and self.postprocessor is not None:
            self.fit_postprocessor_on_domain(self.domain)

    def fit_postprocessor_on_domain(self, domain):
        values = self.score_samples(domain, no_postprocess=True)
        self.postprocessor.fit(values)

    def score_samples(self, test_points, no_postprocess=False):
        scores = self.gp_model(test_points).stddev.detach().unsqueeze(-1).pow(2)
        if self.loggify:
            scores = scores.log()
        if no_postprocess:
            return scores
        if isinstance(self.postprocessor, MinMaxScaler):
            scores = self.postprocessor.transform(scores.numpy())
        values = torch.FloatTensor(scores)
        assert values.ndim == 2 and values.size(0) == len(test_points)
        return values
