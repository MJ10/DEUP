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

    def __call__(self, x):
        # x should be a n x d tensor
        features = []
        if 'x' in self.features:
            features.append(x)
        if 'd' in self.features:
            density_feature = self.density_estimator.score_samples(x.cpu())
            features.append(density_feature)
        if 'D' in self.features:
            distance_feature = self.distance_estimator.score_samples(x.cpu())
            features.append(distance_feature)
        if 'v' in self.features:
            try:
                variance_feature = self.variance_estimator.score_samples(x.cpu())
            except NotPSDError:  # covariance matrix predicted by GP is not PSD
                print('NotPSDError, using zeros as variance feature for this step')
                variance_feature = torch.zeros((x.size(0), 1))
            features.append(variance_feature)
        if 'b' in self.features:
            if self.training_set is not None:
                distance_to_train = torch.min(torch.cdist(x, self.training_set), dim=1, keepdim=True).values
                binary_feature = (distance_to_train < self.epsilon).to(torch.float32)
            else:
                binary_feature = torch.zeros((x.size(0), 1))
            features.append(binary_feature)
        return torch.cat(features, dim=1)
