from .density_estimator import (FixedKernelDensityEstimator, CVKernelDensityEstimator, MAFMOGDensityEstimator,
                                MADEMOGDensityEstimator, DistanceEstimator, VarianceSource)
from .uncertainty_estimation_utils import get_dropout_uncertainty_estimate, get_ensemble_uncertainty_estimate
from .networks import create_network, create_optimizer, create_multiplicative_scheduler, reset_weights
from .maf import MAFMOG

from .test_functions import sinusoid, multi_optima, booth, levi_n13, ackley200, ackley10
from .logging import hash_args, log_args, log_results, compute_exp_dir

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
          'levi_n13': (2, (-4, 4)),
          'ackley200': (200, (-5, 10)),
          'ackley10': (10, (-5, 10))
          }
