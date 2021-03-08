from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from uncertaintylearning.features.density_estimator import MAFMOGDensityEstimator
from uncertaintylearning.features.variance_estimator import DUQVarianceSource
from uncertaintylearning.utils import create_network
from uncertaintylearning.models import DEUP
from uncertaintylearning.utils.resnet import ResNet18plus
from uncertaintylearning.features.feature_generator import FeatureGenerator
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats

parser = ArgumentParser()

parser.add_argument("--load_base_path", default='.',
                    help='path to load models')

parser.add_argument("--data_base_path", default='data',
                    help='path to load datasets')

parser.add_argument("--features", default='bvd',
                    help="features to use for training. combination of [d (desnity), v(variance), b(bit), x]. eg \'dvb\'")

args = parser.parse_args()

save_base_path = args.load_base_path
data_base_path = args.data_base_path
features = args.features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_generator(features, density_estimator, variance_estimator):
    return FeatureGenerator(features, density_estimator=density_estimator, variance_estimator=variance_estimator)

def test_ood(model, iid_loader, ood_loader):
    scores = []
    losses = []
    loss_fn = nn.BCELoss(reduction='none')

    # for inp, label in iid_loader:
    #     with torch.no_grad():
    #         score = torch.exp(model._epistemic_uncertainty(inp, is_unseen=True))
    #         loss = loss_fn(model.f_predictor(inp.to(device)), F.one_hot(label, 10).float().to(device)).sum(1).view(-1)
    #     losses.extend(loss.cpu().numpy().tolist())

    for inp, label in ood_loader:
        with torch.no_grad():
            score = torch.exp(model._uncertainty(x=inp, unseen=True))
            loss = loss_fn(model.f_predictor(inp.to(device)), F.one_hot(label, 10).float().to(device)).sum(1).view(-1)

        losses.extend(loss.cpu().numpy().tolist())

    print("Ranked Correlation: {}".format(scipy.stats.spearmanr(scores, losses)))

def get_split_dataset(split_num, dataset):
    idx = torch.logical_or(torch.tensor(dataset.targets) == splits[split_num][0],
                           torch.tensor(dataset.targets) == splits[split_num][1])
    return torch.utils.data.dataset.Subset(dataset, np.where(idx == 0)[0])

splits = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

oodset = torchvision.datasets.SVHN(root=data_base_path, split='test',
                                   download=True, transform=test_transform)
oodloader = torch.utils.data.DataLoader(oodset, batch_size=128,
                                        shuffle=False, num_workers=2)

iid_testset = torchvision.datasets.CIFAR10(root=data_base_path, train=False,
                                           download=True, transform=test_transform)
iid_testloader = torch.utils.data.DataLoader(iid_testset, batch_size=128,
                                             shuffle=False, num_workers=2)

trainset = torchvision.datasets.CIFAR10(root=data_base_path, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

density_estimator = MAFMOGDensityEstimator(n_components=10, hidden_size=1024, batch_size=100, n_blocks=5, lr=1e-4,
                                           use_log_density=True, epochs=32, use_density_scaling=True)
density_estimator.fit(trainset, device, "", init_only=True)
variance_source = DUQVarianceSource(32, 10, 512, 512, 0.1, 0.999, 0.5, device)

networks = {
    'e_predictor': create_network(len(features), 1, 1024, 'relu', False, 5),
    'f_predictor': ResNet18plus()  # use create_wrapped_network("resnet50") for resnet-50
}

optimizers = {
    'e_optimizer': optim.SGD(networks['e_predictor'].parameters(), lr=0.005, momentum=0.9),
    'f_optimizer': optim.SGD(networks['f_predictor'].parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
}

data = {
    'dataloader': trainloader,
}

model = DEUP(data=data,
            feature_generator=None,
            networks=networks,
            optimizers=optimizers,
            device=device,
            loss_fn=nn.BCELoss(reduction='none'),
            e_loss_fn=nn.MSELoss(),
            batch_size=256
        )

model = model.to(device)

ood_data_x, ood_data_y = [], []

for split_num in range(0):
    density_save_path = save_base_path + "mafmog_cifar_split_{}_new.pt".format(split_num)
    density_estimator.model.load_state_dict(torch.load(density_save_path))
    density_estimator.model.to(device)
    density_estimator.postprocessor.fit(
        density_estimator.score_samples(trainset, device, no_preprocess=True))

    var_save_path = save_base_path + "duq_cifar_split_{}_new.pt".format(split_num)
    variance_source.model = torch.load(var_save_path)
    variance_source.model.to(device)
    variance_source.postprocessor.fit(
        variance_source.score_samples(loader=trainloader, no_preprocess=True))

    model_save_path = save_base_path + "resnet18_cifar_split_{}_new.pt".format(split_num)
    model.f_predictor = torch.load(model_save_path)

    feature_generator = get_feature_generator(features, density_estimator, variance_source)
    
    epi_x, epi_y = feature_generator.build_dataset(model.f_predictor, trainloader, 
        nn.BCELoss(reduction='none'), splits[split_num], device
    )
    ood_data_x.append(epi_x)
    ood_data_y.append(epi_y)

# ood_x_set = torch.cat(ood_data_x, dim=0)
# ood_y_set = torch.cat(ood_data_y, dim=0)

# torch.save(ood_y_set, save_base_path+"y_18_" + features + ".pt")
# torch.save(ood_x_set, save_base_path+"x_18_" + features +".pt")

ood_y_set = torch.load(save_base_path + "y_18_" + features + ".pt")
ood_x_set = torch.load(save_base_path + "x_18_" + features + ".pt")

density_save_path = save_base_path + "mafmog_cifar_full_new.pt"
density_estimator.model.load_state_dict(torch.load(density_save_path))
density_estimator.model.to(device)
density_estimator.postprocessor.fit(density_estimator.score_samples(trainset, device, no_preprocess=True))

var_save_path = save_base_path + "duq_cifar_full_new.pt"
variance_source.mode = torch.load(var_save_path)
variance_source.model.to(device)
variance_source.postprocessor.fit(variance_source.score_samples(loader=trainloader, no_preprocess=True))

model_save_path = save_base_path + "resnet18_cifar_full_new.pt"
model.f_predictor = torch.load(model_save_path)

feature_generator = get_feature_generator(features, density_estimator, variance_source)
model.feature_generator = feature_generator
model.feature_generator.density_estimator.model.to(device)
model.feature_generator.variance_estimator.model.to(device)

model.fit_uncertainty_estimator(ood_x_set, ood_y_set, epochs=1)
model = model.to(device)
test_ood(model, iid_testloader, oodloader)
