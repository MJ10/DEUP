import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from .density_estimator import VarianceSource
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

class DUQVarianceSource(VarianceSource):
    def __init__(self, input_size, num_classes, centroid_size, model_output_size, length_scale, gamma, l_gradient_penalty, device):
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
                        targets= targets.to(self.device)
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets.to(self.device)).sum().item()
                acc = 100.*correct/total
                print("Epoch: {}, test acc: {}, test loss {}".format(epoch, acc, test_loss / total))

            self.scheduler.step()

        if save_path is not None:
            torch.save(self.model, save_path)

    def score_samples(self, data=None, loader=None, no_preprocess=False):
        self.model.eval()

        with torch.no_grad():
            scores = []
            
            if loader is None:
                data=data.to(self.device)
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