import numpy as np

from scipy.stats import invwishart
from scipy.stats import invgamma

class ImproperUniform:

    def __init__(self, lower_bound=None, upper_bound=None):
        self._lower = lower_bound
        self._upper = upper_bound

        if self._lower is None:
            self._lower = -np.inf

        if self._upper is None:
            self._upper = np.inf

    def pdf(self, x):
        '''
        :param x: input array
        :type x: 1D or 2D array; if 2D, must squeeze to 1D
        '''
        array_x = np.array(x).squeeze()

        if array_x.ndim > 1:
            raise ValueError('Input array must be 1D or must squeeze to 1D')

        prior_pdf = np.zeros(array_x.size)
        in_bounds = (array_x >= self._lower) & (array_x <= self._upper)
        return np.add(prior_pdf, 1, out=prior_pdf, where=in_bounds)


class InvGamma:

    def __init__(self, alpha=1, beta=1):

        self._alpha = alpha
        self._beta = beta
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta

    def pdf(self, x):
        
        array_x = np.array(x).squeeze()

        if array_x.ndim > 1:
            raise ValueError('Input array must be 1D or must squeeze to 1D')

        return invgamma.pdf(array_x, a=self.alpha, loc=1./self.beta)


class InvWishart:

    def __init__(self, dof, scale):
        '''
        :param scale: scale matrix
        :param dof: degrees of freedom (if == scale.shape[0], noninformative)
        '''
        self._cov_dim = scale.shape[0]
        self._dim = int(self._cov_dim * (self._cov_dim + 1) / 2)
        self._invwishart = invwishart(dof, scale)

    @property
    def dim(self):
        return self._dim

    def rvs(self, num_samples):
        '''
        :param num_samples: number of samples to return
        :type num_samples: int
        '''
        cov = self._invwishart.rvs(num_samples)
        idx1, idx2 = np.triu_indices(self._cov_dim)
        return cov[:, idx1, idx2]

    def pdf(self, x):
        '''
        :param x: input array
        :type x: 2D array (# samples, # unique covariances)
        '''
        covs = self._assemble_covs(x)
        try:
            return self._invwishart.pdf(covs)
        except np.linalg.LinAlgError:
            return np.zeros((x.shape[0], 1))

    def _assemble_covs(self, samples):
        covs = np.zeros((samples.shape[0], self._cov_dim, self._cov_dim))
        idx1, idx2 = np.triu_indices(self._cov_dim)
        covs[:, idx1, idx2] = samples
        covs += np.transpose(np.triu(covs, 1), axes=(0, 2, 1))
        return np.transpose(covs, axes=(1, 2, 0))
