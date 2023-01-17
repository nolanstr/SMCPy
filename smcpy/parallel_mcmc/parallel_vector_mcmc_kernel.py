from copy import copy
import numpy as np

from .parallel_kernel_base import ParallelMCMCKernel

class ParallelVectorMCMCKernel(ParallelMCMCKernel):

    def __init__(self, vector_mcmc_object, param_order):
        self._mcmc = vector_mcmc_object
        self._param_order = param_order

    def mutate_particles(self, params, log_likes, num_samples, cov, phi):
        params, log_likes = self._mcmc.smc_metropolis(params,
                                                           num_samples,
                                                           cov, phi)
        return params, log_likes

    def sample_from_prior(self, num_samples):
        return self._mcmc.sample_from_priors(num_samples)


    def get_log_likelihoods(self, params):
        return self._mcmc.evaluate_log_likelihood(params)

    def get_log_priors(self, params):
        if params.ndim == 2:
            log_priors = self._mcmc.evaluate_log_priors(
                                            params[np.newaxis,:,:])
        else:
            log_priors = self._mcmc.evaluate_log_priors(params)
        return np.sum(log_priors, axis=1).reshape(-1, 1)
