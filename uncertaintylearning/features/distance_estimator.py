import torch


class DistanceEstimator:
    def __init__(self):
        self.dim = None
        self.training_points = None

    def fit(self, training_points):
        self.dim = training_points.shape[1]
        self.training_points = training_points

    def score_samples(self, test_points):
        values = torch.cdist(self.training_points, test_points).min(axis=0)[0]
        values = torch.FloatTensor(values).unsqueeze(-1)
        assert values.ndim == 2 and values.size(0) == len(test_points)
        return values