'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
'''

import numpy as np

from abc import ABC, abstractmethod

from .parallel_smc.initializer import Initializer
from .parallel_smc.mutator import Mutator


class ParallelSamplerBase:

    def __init__(self, mcmc_kernel):
        self._mcmc_kernel = mcmc_kernel
        self._initializer = Initializer(self._mcmc_kernel)
        self._mutator = Mutator(self._mcmc_kernel)
        self._updater = None
        self._mutation_ratio = 1

    @abstractmethod
    def sample(self):
        '''
        Performs SMC sampling. Returns step list and estimates of marginal
        log likelihood at each step.
        '''

    def _initialize(self, num_particles, proposal, parallel):
        if proposal is None:
            return self._initializer.init_particles_from_prior(num_particles,
                    parallel)
        else:
            return self._initializer.init_particles_from_samples(*proposal,
                    parallel)

    def _do_smc_step(self, particles, phi, delta_phi, num_mcmc_samples):
        particles = self._updater.update(particles, delta_phi)
        mut_particles = self._mutator.mutate(particles, phi,
                                                num_mcmc_samples)
        self._compute_mutation_ratio(particles, mut_particles)
        return mut_particles

    def _compute_mutation_ratio(self, old_particles, new_particles):
        mutated = ~np.all(new_particles.params == old_particles.params, axis=1)
        self._mutation_ratio = sum(mutated) / new_particles.params.shape[0]

    def _estimate_marginal_log_likelihoods(self):
        sum_un_log_wts = [np.zeros(len(self.steps))] + [self._logsum(ulw) \
                          for ulw in self._updater._unnorm_log_weights]
        num_updates = len(sum_un_log_wts[0])
        mlls = np.cumsum(sum_un_log_wts, axis=0) 
        return mlls.T

    @staticmethod
    def _logsum(Z):
        Z = -np.sort(-Z, axis=0) # descending
        Z0 = Z[0, :]
        Z_shifted = Z[1:, :] - Z0
        return Z0 + np.log(1 + np.sum(np.exp(Z_shifted), axis=0))
