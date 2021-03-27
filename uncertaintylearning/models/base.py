import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal


class BaseModel(Model):
    """
    Base class for all models implemented in the repo
    """

    def post_forward(self, means, variances):
        # TODO: maybe the two cases can be merged into one with torch.diag_embed
        assert means.ndim == variances.ndim
        if means.ndim == 2:
            mvn = MultivariateNormal(means.squeeze(), torch.diag(variances.squeeze() + 1e-6))
        elif means.ndim == 3:
            assert means.size(-1) == variances.size(-1) == 1
            try:
                mvn = MultivariateNormal(means.squeeze(-1), torch.diag_embed(variances.squeeze(-1) + 1e-6))
            except RuntimeError:
                print('RuntimeError')
                print(torch.diag_embed(variances.squeeze(-1)) + 1e-6)
        else:
            raise NotImplementedError("Something is wrong, just cmd+f this error message and you can start debugging.")
        return mvn

    @property
    def num_outputs(self):
        return self.output_dim

    def get_prediction_with_uncertainty(self, x, **kwargs):
        if x.ndim == 3:
            assert len(kwargs) == 0, "no kwargs can be given if x.ndim == 3"
            preds = self.get_prediction_with_uncertainty(x.view(x.size(0) * x.size(1), x.size(2)), **kwargs)
            return preds[0].view(x.size(0), x.size(1), 1), preds[1].view(x.size(0), x.size(1), 1)

    def posterior(self, x, **kwargs):
        mvn = self.forward(x, **kwargs)
        return GPyTorchPosterior(mvn)

    def forward(self, x, **kwargs):
        means, variances = self.get_prediction_with_uncertainty(x, **kwargs)
        return self.post_forward(means, variances)
