import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from .density_estimator import VarianceSource

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
        self.model = ResNet_DUQ(input_size, num_classes, centroid_size, model_output_size, length_scale, gamma)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[25, 50, 75], gamma=0.2
        )
    
    def fit(self, epochs=75, train_loader=None, save_path=None):
        
        for epoch in tqdm(range(epochs)):
            running_loss = 0
            for i, (x, y) in enumerate(train_loader):
                self.model.train()

                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                if self.l_gradient_penalty > 0:
                    x.requires_grad_(True)

                z, y_pred = self.model(x)
                y = F.one_hot(y, num_classes).float()

                loss = bce_loss_fn(y_pred, y)

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

            self.scheduler.step()
            
        if save_path is not None:
            torch.save(self.model, save_path)

    def score_samples(self, loader):
        pass