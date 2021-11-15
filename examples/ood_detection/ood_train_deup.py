from matplotlib import pyplot as plt
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from uncertaintylearning.features.density_estimator import MAFMOGDensityEstimator
from uncertaintylearning.features.variance_estimator import DUEVarianceSource
from uncertaintylearning.utils import create_network
from uncertaintylearning.models import DEUP
from uncertaintylearning.utils.resnet import ResNet18plus
from uncertaintylearning.features.feature_generator import FeatureGenerator
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import scipy.stats
import os
from PIL import Image

parser = ArgumentParser()

parser.add_argument("--load_base_path", default='.',
                    help='path to load models')

parser.add_argument("--data_base_path", default='data',
                    help='path to load datasets')

parser.add_argument("--features", default='bvd',
                    help="features to use for training. combination of [d (desnity), v(variance), b(bit), x]. eg \'dvb\'")

parser.add_argument("--ood_set", default='SVHN',
                    help="OOD dataset to use for eval. CIFAR10C or SVHN")

parser.add_argument("--corruption_type", default='gaussian_noise',
                    help="OOD dataset to use for eval")

parser.add_argument("--cifar10c_path", default='data/',
                    help="path for cifar10c")

args = parser.parse_args()

save_base_path = args.load_base_path
data_base_path = args.data_base_path
features = args.features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corruptions = [
    "gaussian_noise",
    "shot_noise",
    "speckle_noise",
    "impulse_noise",
    "defocus_blur",
    "gaussian_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "spatter",
    "saturate",
    "frost"
]

class CIFAR10C(torchvision.datasets.VisionDataset):
    def __init__(self, root, name, transform=None, target_transform=None):
        assert name in corruptions
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)


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

if args.ood_set == "SVHN":
    oodset = torchvision.datasets.SVHN(root=data_base_path, split='test',
                                    download=True, transform=test_transform)
elif args.ood_set == "CIFAR10C":
    oodset = CIFAR10C(args.cifar10c_path, 
                                    args.corruption_type, transform=test_transform)
else:
    print("Please select valid dataset")
    exit(0)

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
variance_source = DUEVarianceSource(32, 10, True, 1, 0.99,
                50, 0.05, 5e-4, None, 'RBF', False, False, 2, device)
variance_source.fit(train_loader=trainloader, save_path=None, epochs=0)

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

for split_num in range(len(splits)):
    print(split_num)
    density_save_path = save_base_path + "mafmog_cifar_split_{}_new.pt".format(split_num)
    density_estimator.model.load_state_dict(torch.load(density_save_path))
    density_estimator.model.to(device)
    density_estimator.postprocessor.fit(
        density_estimator.score_samples(trainset, device, no_preprocess=True))

    var_save_path = save_base_path + "due_cifar_split_{}_new_".format(split_num)
    variance_source.load(var_save_path)
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

ood_x_set = torch.cat(ood_data_x, dim=0)
ood_y_set = torch.cat(ood_data_y, dim=0)

# torch.save(ood_y_set, save_base_path+"y_18_" + features + ".pt")
# torch.save(ood_x_set, save_base_path+"x_18_" + features +".pt")

# ood_y_set = torch.load(save_base_path + "y_18_" + features + ".pt")
# ood_x_set = torch.load(save_base_path + "x_18_" + features + ".pt")

density_save_path = save_base_path + "mafmog_cifar_full_new.pt"
density_estimator.model.load_state_dict(torch.load(density_save_path))
density_estimator.model.to(device)
density_estimator.postprocessor.fit(density_estimator.score_samples(trainset, device, no_preprocess=True))

var_save_path = save_base_path + "due_cifar_full_new_"
variance_source.load(var_save_path)
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
