from matplotlib import pyplot as plt
import numpy as np
import torch
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator,
                                       create_network, create_optimizer, create_multiplicative_scheduler, create_wrapped_network)
from uncertaintylearning.models import DEUP

from torchvision import datasets, models, transforms as T

device=torch.device("cuda" if torch.cuda.is_cuda() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

iid_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
iid_testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

oodset = torchvision.datasets.SVHN(root='./data', train=False,
                                       download=True, transform=transform)
oodloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

testset = torchvision.datasets.ImageNet(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


networks = {
            'a_predictor': create_network(1, 1, 32, 'relu', True),
            'e_predictor': create_network(1, 1, 32, 'relu', True),
            'f_predictor': create_wrapped_network("resnet50", num_classes=10)
            }

optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], 1e-2),
              'e_optimizer': create_optimizer(networks['e_predictor'], 3e-3),
              'f_optimizer': create_optimizer(networks['f_predictor'], 1e-3)
              }

data = {
    'train_loader': trainloader,
    'ood_loader': oodloader
}

density_estimator = MAFMOGDensityEstimator(n_components=10, hidden_size=1024, batch_size=64, n_blocks=5, lr=1e-4, use_log_density=True, use_density_scaling=True)
density_estimator.fit(trainset, device)

model = DEUP(data=data,
            networks=networks,
            optimizers=optimizers,
            density_estimator=density_estimator,
            features='d',
            device=device,
            use_dataloaders=True,
            loss_fn=nn.BCELoss(reduction='sum'),
            batch_size=64
            )

model = model.to(device)

epochs = 100

new_losses = model.fit(epochs=epochs)
for key in 'afe':
    losses[key].extend(new_losses[key])
