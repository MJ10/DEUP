from .density_estimator import FixedKernelDensityEstimator, CVKernelDensityEstimator
from .dropout_utils import get_uncertainty_estimate
from .networks import create_network, create_optimizer, create_multiplicative_scheduler, reset_weights

from .test_functions import sinusoid, multi_optima, booth, levi_n13
from .logging import hash_args, log_args, log_results, compute_exp_dir

functions = {'sinusoid': sinusoid,
             'multi_optima': multi_optima,
             'booth': booth,
             'levi_n13': levi_n13
             }

bounds = {'sinusoid': (1, (-1, 2)),
          'multi_optima': (1, (-1, 2)),
          'booth': (2, (-4, 4)),
          'levi_n13': (2, (-4, 4))
          }
