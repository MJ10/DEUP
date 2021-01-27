from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator, MAFMOGDensityEstimator,
                                       create_network, create_optimizer, create_multiplicative_scheduler, create_wrapped_network)
from uncertaintylearning.models import DEUP
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_ood(self, model, iid_loader, ood_loader):
    scores = []
    labels = []
    
    for inp, _ in iid_loader:
        with torch.no_grad():
            _, score = model.get_prediction_with_uncertainty(inp)
        
        scores.extend([score.cpu().numpy().tolist()])
        labels.extend([0 for _ in range(inp.size(0))])
    
    for inp, _ in ood_loader:
        with torch.no_grad():
            _, score = model.get_prediction_with_uncertainty(inp)
        
        scores.extend([score.cpu().numpy().tolist()])
        labels.extend([1 for _ in range(inp.size(0))])
    
    roc_auc = roc_auc_score(y_true=labels, y_score=scores)

    print("OOD ROC AUC: {}".format(roc_auc))

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/network/tmp1/moksh.jain/data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

iid_testset = torchvision.datasets.CIFAR10(root='/network/tmp1/moksh.jain/data', train=False,
                                       download=True, transform=transform)
iid_testloader = torch.utils.data.DataLoader(iid_testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# oodset = torchvision.datasets.SVHN(root='/network/tmp1/moksh.jain/data', split='test',
#                                          download=True, transform=transform)
# oodloader = torch.utils.data.DataLoader(oodset, batch_size=64,
#                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='/network/tmp1/moksh.jain/data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=2)

# Train Density estimator on train set
density_estimator = MAFMOGDensityEstimator(n_components=10, hidden_size=1024, batch_size=256, n_blocks=5, lr=1e-4, use_log_density=True, use_density_scaling=True)
density_estimator.fit(trainset, device)

networks = {
            'a_predictor': create_network(1, 1, 32, 'relu', True),
            'e_predictor': create_network(1, 1, 32, 'relu', True),
            'f_predictor': create_wrapped_network("resnet50", num_classes=10)
            }

optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], 1e-2),
              'e_optimizer': create_optimizer(networks['e_predictor'], 3e-3),
              'f_optimizer': optim.Adam(networks['f_predictor'].parameters(), 1e-3)
              }

data = {
    'train_loader': trainloader,
    'ood_loader': oodloader
}

model = DEUP(data=data,
            networks=networks,
            optimizers=optimizers,
            density_estimator=density_estimator,
            features='d',
            device=device,
            use_dataloaders=True,
            loss_fn=nn.BCELoss(reduction='none'),
            batch_size=128
            )

model = model.to(device)

epochs = 100
new_losses = model.fit(epochs=epochs, val_loader=iid_testloader)
model.fit_ood(epochs=epochs)
