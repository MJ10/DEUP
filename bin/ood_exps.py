from matplotlib import pyplot as plt
import numpy as np
import torch
from uncertaintylearning.utils import (FixedKernelDensityEstimator, CVKernelDensityEstimator,
                                       create_network, create_optimizer, create_multiplicative_scheduler)
from uncertaintylearning.models import EpistemicPredictor

from torchvision import datasets, models, transforms as T

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# TODO: implement create_conv_network
networks = {'a_predictor': create_conv_network(1, 1, 32, 'relu', True),
            'e_predictor': create_conv_network(1, 1, 32, 'relu', True),
            'f_predictor': models.ResNet50(pretrained=False, num_classes=10)
            }

optimizers = {'a_optimizer': create_optimizer(networks['a_predictor'], 1e-2),
              'e_optimizer': create_optimizer(networks['e_predictor'], 3e-3),
              'f_optimizer': create_optimizer(networks['f_predictor'], 1e-3)
              }

density_estimator = None # add flow/vae model

model = EpistemicPredictor(train_X=x,
                           train_Y=y,
                           networks=networks,
                           optimizers=optimizers,
                           density_estimator=density_estimator,
                           train_Y_2=y2,
                           ood_X=ood_x,
                           ood_Y=ood_y)

epochs = 2
losses = {'a': [], 'e': [], 'f': []}

for i in range(epochs):
    new_losses = model.fit()
    for key in 'afe':
        losses[key].extend(new_losses[key])


plt.plot(losses['f'], label='f_loss')
plt.plot(losses['a'], label='a_loss')
plt.plot(losses['e'], label='u_loss')
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
predictions = networks['f_predictor'](x_test).detach().numpy().ravel()
aleatoric_u = networks['a_predictor'](x_test).detach().numpy().ravel()
a_low = predictions - np.sqrt(aleatoric_u)
a_high = predictions + np.sqrt(aleatoric_u)

plt.plot(x_test, y_test, '.', label='test_samples', alpha=.15)
plt.plot(x_test, predictions, label='predictions', alpha=.3, lw=2)
plt.fill_between(x_test.numpy().ravel(), a_low, a_high, alpha=.3, label='aleatoric')

test_den = torch.FloatTensor(density_estimator.score_samples(x_test.numpy()).reshape(-1, 1))
e_in = torch.cat((x_test.view(-1, 1), test_den), axis=1)
epistemic_u = networks['e_predictor'](e_in).detach().numpy().ravel()

e_low = predictions - np.sqrt(epistemic_u)
e_high = predictions + np.sqrt(epistemic_u)

plt.fill_between(x_test.numpy().ravel(), e_low, e_high, alpha=.3, label='epistemic')

exp_epistemic_uncertainty = (networks['f_predictor'](x_test) - y_test).pow(2).detach().numpy().ravel()
exp_total_uncertainty = exp_epistemic_uncertainty + noise ** 2

exp_e_low = predictions - np.sqrt(exp_epistemic_uncertainty)
exp_e_high = predictions + np.sqrt(exp_epistemic_uncertainty)

plt.fill_between(x_test.numpy().ravel(), exp_e_low, exp_e_high, alpha=.15, label='out of sample error')

plt.legend()
plt.show()

total_uncertainty = epistemic_u + aleatoric_u

plt.plot(x_test, exp_total_uncertainty, label='true total uncertainty')
plt.plot(x_test, total_uncertainty, label='predicted total uncertainty')
plt.legend()
plt.show()
