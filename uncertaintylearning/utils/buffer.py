import torch


class Buffer:
    def __init__(self, input_dim, features='xd'):
        self.targets = torch.empty((0, 1))
        self.features = torch.empty((0, len(features) + (input_dim - 1 if 'x' in features else 0)))

    def add_features(self, features):
        self.features = torch.cat((self.features, features))

    def add_targets(self, targets):
        self.targets = torch.cat((self.targets, targets))

    def __repr__(self):
        return "Buffer with {} elements".format(self.targets.size(0))