import torch
from gpytorch.utils.errors import NotPSDError


class FeatureGenerator:
    def __init__(self, features, density_estimator=None, distance_estimator=None, variance_estimator=None,
                 training_set=None, epsilon=1e-3):
        # Features is a string containing one or some of the characters 'xdDvb'
        # If training_set is not None (should be a n x d tensor), then it is used to define the binary feature
        self.features = features
        if 'd' in features:
            assert density_estimator is not None
            self.density_estimator = density_estimator
        if 'v' in features:
            assert variance_estimator is not None
            self.variance_estimator = variance_estimator
        if 'D' in features:
            assert distance_estimator is not None
            self.distance_estimator = distance_estimator
        if 'b' in features:
            self.training_set = training_set
            self.epsilon = epsilon

    def __call__(self, x, unseen=True, device=torch.device("cpu")):
        # x should be a n x d tensor
        features = []
        if 'x' in self.features:
            features.append(x)
        if 'd' in self.features:
            density_feature = self.density_estimator.score_samples(x.to(device)).to(device)
            features.append(density_feature)
        if 'D' in self.features:
            distance_feature = self.distance_estimator.score_samples(x.cpu())
            features.append(distance_feature)
        if 'v' in self.features:
            try:
                variance_feature = self.variance_estimator.score_samples(x.to(device))
            except NotPSDError:  # covariance matrix predicted by GP is not PSD
                print('NotPSDError, using zeros as variance feature for this step')
                variance_feature = torch.zeros((x.size(0), 1))
            features.append(variance_feature)
        if 'b' in self.features:
            if self.training_set is not None:
                distance_to_train = torch.min(torch.cdist(x, self.training_set), dim=1, keepdim=True).values
                binary_feature = (distance_to_train < self.epsilon).to(torch.float32)
            else:
                if unseen:
                    binary_feature = torch.zeros((x.size(0), 1))
                else:
                    binary_feature = torch.ones((x.size(0), 1))
            features.append(binary_feature)
        return torch.cat(features, dim=1)

    def build_dataset(self, model, loader, loss_fn, classes, device, num_classes=10):
        epistemic_x = []
        epistemic_y = []

        with torch.no_grad():
            for x, y in loader:
                u_out = torch.clamp(torch.log10(
                    loss_fn(model(x.to(device)), torch.nn.functional.one_hot(y, num_classes).to(device, torch.float32)).sum(1)),
                                    min=-5).view(-1, 1)
                u_in = []
                if 'd' in self.features:
                    density_feature = self.density_estimator.score_samples(x, device, no_preprocess=False).to(device)
                    u_in.append(density_feature)
                if 'v' in self.features:
                    variance_feature = self.variance_estimator.score_samples(data=x, no_preprocess=False).to(device).view(-1, 1)
                    u_in.append(variance_feature)
                if 'b' in self.features:
                    u_in.append(torch.logical_or(y == classes[0], y == classes[1]).view(-1, 1).float().to(device))
                u_in = torch.cat(u_in, dim=1)
                epistemic_x.append(u_in)
                epistemic_y.append(u_out)
            epi_x, epi_y = torch.cat(epistemic_x, dim=0), torch.cat(epistemic_y, dim=0)
        return epi_x, epi_y