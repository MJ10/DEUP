from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator, MAFMOGDensityEstimator, DUQVarianceSource,
                                       create_network, create_optimizer, create_multiplicative_scheduler, create_wrapped_network)
from uncertaintylearning.models import DEUP
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = "/network/tmp1/moksh.jain/models/"

def test_ood(model, iid_loader, ood_loader):
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

splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)] 

def get_split_dataset(split_num, dataset):
    # import pdb;pdb.set_trace()
    idx = torch.logical_or(torch.tensor(dataset.targets)==splits[split_num][0], torch.tensor(dataset.targets)==splits[split_num][1])
    print(idx.sum())
    # dataset.targets = dataset.targets[idx]
    # dataset.data = dataset.data[idx]
    return torch.utils.data.dataset.Subset(dataset, np.where(idx==0)[0])

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

oodset = torchvision.datasets.SVHN(root='/network/tmp1/moksh.jain/data', split='test',
                                         download=True, transform=transform)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=64,
                                         shuffle=False, num_workers=2)

# testset = torchvision.datasets.CIFAR100(root='/network/tmp1/moksh.jain/data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128,
#                                          shuffle=False, num_workers=2)

iid_testset = torchvision.datasets.CIFAR10(root='/network/tmp1/moksh.jain/data', train=False,
                                    download=True, transform=transform)
iid_testloader = torch.utils.data.DataLoader(iid_testset, batch_size=128,
                                        shuffle=False, num_workers=2)

trainset = torchvision.datasets.CIFAR10(root='/network/tmp1/moksh.jain/data', train=True,
                                    download=True, transform=transform)
# trainset = get_split_dataset(split_num, dataset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                        shuffle=True, num_workers=2)

density_estimator = MAFMOGDensityEstimator(n_components=10, hidden_size=1024, batch_size=100, n_blocks=5, lr=1e-4, use_log_density=True, epochs=50, use_density_scaling=True)
variance_source = DUQVarianceSource(32, 10, 512, 512, 0.1, 0.999, 0.5, device)


networks = {
            'a_predictor': create_network(1, 1, 32, 'relu', True),
            'e_predictor': create_network(1, 1, 32, 'relu', True),
            'f_predictor': create_wrapped_network("resnet50", num_classes=10)
            }

optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], 1e-2),
            'e_optimizer': create_optimizer(networks['e_predictor'], 3e-3),
            'f_optimizer': optim.SGD(networks['f_predictor'].parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            }
schedulers = {
    'f_scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['f_optimizer'], T_max=200)
}
data = {
    'train_loader': trainloader,
    'ood_loader': oodloader
}

model = DEUP(data=data,
            networks=networks,
            optimizers=optimizers,
            density_estimator=density_estimator,
            variance_source=variance_source,
            features='xdv',
            device=device,
            use_dataloaders=True,
            loss_fn=nn.BCELoss(reduction='none'),
            batch_size=128
        )

model = model.to(device)
density_save_path = base_path + "mafmog_cifar_full.pt"
density_estimator.fit(trainset, device, density_save_path)

var_save_path = base_path + "duq_cifar_full.pt"
variance_source.fit(train_loader=trainloader, save_path=var_save_path)

model_save_path = base_path + "resnet_cifar_full.pt"
model.f_predictor.load_state_dict(torch.load(model_save_path))
# epochs = 200
model.fit(epochs=epochs, val_loader=iid_testloader)

# for split_num in range(len(splits)):
    # print(split_num)
    # density_save_path = base_path + "mafmog_cifar_split_{}.pt".format(split_num)
    # # Train Density estimator on train set
    # # model.density_estimator.model.load_state_dict(torch.load(density_save_path))
    # density_estimator.fit(trainset, device, density_save_path)

    # var_save_path = base_path + "duq_cifar_split_{}.pt".format(split_num)
    # variance_source.fit(train_loader=trainloader, save_path=var_save_path)
    # # model.variance_source.model.load_state_dict(torch.load(var_save_path))

    # model_save_path = base_path + "resnet_cifar_split_{}.pt".format(split_num)
    # model.f_predictor.load_state_dict(torch.load(model_save_path))
    # model.fit_ood(epochs=epochs, loader=trainloader)
