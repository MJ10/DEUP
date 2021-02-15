from .density_estimator import (FixedKernelDensityEstimator, CVKernelDensityEstimator, MAFMOGDensityEstimator,
                                MADEMOGDensityEstimator, DistanceEstimator, VarianceSource, FixedSmoothKernelDensityEstimator)
from .uncertainty_estimation_utils import get_dropout_uncertainty_estimate, get_ensemble_uncertainty_estimate
from .networks import create_network, create_optimizer, reset_weights, create_wrapped_network, create_epistemic_pred_network
from .maf import MAFMOG
from .variance_estimator import DUQVarianceSource

from .test_functions import sinusoid, multi_optima, booth, levi_n13, ackley200, ackley10
import torch

functions = {'sinusoid': sinusoid,
             'multi_optima': multi_optima,
             'booth': booth,
             'levi_n13': levi_n13,
             'ackley200': ackley200,
             'ackley10': ackley10,
             }

bounds = {'sinusoid': (1, (-1, 2)),
          'multi_optima': (1, (-1, 2)),
          'booth': (2, (-4, 4)),
          'levi_n13': (2, (-10, 10)),
          'ackley200': (200, (-5, 10)),
          'ackley10': (10, (-5, 10))
          }


def softplus(x, beta=1):
    return 1. / beta * (torch.log((beta * x).exp() + 1))

