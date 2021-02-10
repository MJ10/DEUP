import numpy as np

from .smooth_kde import SmoothKDE

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


def remove(x, idx):
    # x is a n x ... array, this returns a (n - 1) x ... array by removing the i-th element/row/...
    return x[np.arange(len(x)) != idx]


class CrossValidator:
    """
    This class picks a kernel density estimator amongst a pre-determined set, based on their predictive performances
    on the generalization gap of a Kernel Regressor.
    """

    def __init__(self, x, y, kernel='gaussian', bandwidth_search_space=None, M=10, reg=0.5, alpha_search_space=None):
        # x and y should be n x d and n x 1 numpy arrays respectively
        # if bandwidth_search_space is specified (1d numpy array), M is ignored
        # alpha search space is to look for the mixing parameter with the Gaussian (0 = full gaussian, 1 = full KDE)
        # if alpha_search_space is not defined, then it's set to [1]
        self.n, self.dim = x.shape
        self.x = x
        self.y = y

        if bandwidth_search_space is not None:
            self.bandwidth_search_space = bandwidth_search_space
        else:
            self.bandwidth_search_space = np.logspace(-3, 1, M)

        if alpha_search_space is not None:
            self.alpha_search_space = alpha_search_space
        else:
            self.alpha_search_space = [1.]

        self.M = len(self.bandwidth_search_space)
        self.A = len(self.alpha_search_space)

        self.alpha_reg = reg
        self.kernel = kernel

        self.main_regressor = None
        self.in_sample_errors = None
        self.out_of_sample_errors = None
        self.gamma = None
        self.global_kdes = None
        self.kdes = None

        self.linreg_inputs = None
        self.linreg_targets = None

        self.best_estimator_ = None
        self.best_params_ = {}

    def fit(self):
        self.cross_validate_regressor()
        self.eval_out_of_sample_errors()
        self.make_density_estimators()
        self.make_regression_datasets()
        self.pick_best_kde()
        return self.best_estimator_

    def cross_validate_regressor(self):
        gamma_search_space = 1. / (2 * self.bandwidth_search_space ** 2)
        params = {'gamma': gamma_search_space}
        grid_search = GridSearchCV(KernelRidge(alpha=self.alpha_reg),
                                   params,
                                   cv=self.n,  # leave-one-out cross validation
                                   scoring='neg_mean_squared_error')
        grid_search.fit(self.x, self.y)

        self.main_regressor = grid_search.best_estimator_
        self.gamma = grid_search.best_params_['gamma']
        self.in_sample_errors = (self.main_regressor.predict(self.x) - self.y) ** 2

    def eval_out_of_sample_errors(self):
        regressors = [KernelRidge(gamma=self.gamma).fit(remove(self.x, i), remove(self.y, i)) for i in range(self.n)]
        oose = np.array([(regressors[i].predict(self.x[[i]]) - self.y[[i]]).ravel() ** 2 for i in range(self.n)])
        self.out_of_sample_errors = oose

    def make_density_estimators(self):
        self.global_kdes = [[SmoothKDE(kernel=self.kernel, bandwidth=bandwidth, alpha=alpha).fit(self.x)
                             for alpha in self.alpha_search_space]
                            for bandwidth in self.bandwidth_search_space]
        self.kdes = [[[SmoothKDE(kernel=self.kernel, bandwidth=bandwidth, alpha=alpha).fit(remove(self.x, i))
                       for i in range(self.n)]
                      for alpha in self.alpha_search_space]
                     for bandwidth in self.bandwidth_search_space]

    def make_regression_datasets(self):
        self.linreg_inputs = [[np.concatenate((self.global_kdes[m][a].score_samples(self.x)[:, np.newaxis],
                                               np.array([self.kdes[m][a][i].score_samples(self.x[[i]])
                                                         for i in range(self.n)])
                                               ))
                               for a in range(self.A)]
                              for m in range(self.M)]
        self.linreg_targets = np.concatenate((self.in_sample_errors.ravel(), self.out_of_sample_errors.ravel()))

    def pick_best_kde(self):
        linregs = [[LinearRegression().fit(self.linreg_inputs[m][a], self.linreg_targets)
                    for a in range(self.A)]
                   for m in range(self.M)]
        linreg_errors = [[np.mean((linregs[m][a].predict(self.linreg_inputs[m][a]) - self.linreg_targets) ** 2)
                          for a in range(self.A)]
                         for m in range(self.M)]

        best_m, best_a = np.unravel_index(np.array(linreg_errors).argmin(), np.array(linreg_errors).shape)
        self.best_params_['bandwidth'] = self.bandwidth_search_space[best_m]
        self.best_params_['alpha'] = self.alpha_search_space[best_a]
        self.best_estimator_ = self.global_kdes[best_m][best_a]

