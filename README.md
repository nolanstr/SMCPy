SMCPy - **S**equential **M**onte **C**arlo **S**ampling with **Py**thon 
==========================================================================
Python module for uncertainty quantification using a parallel sequential Monte
Carlo sampler.

Uncertainty quantification (UQ) is essential to provide meaningful and reliable
predictions of real-world system performance. One major obstacle for the
implementation of statistical methods for UQ is the use of expensive
computational models. Classical UQ methods such as Markov chain Monte Carlo
(MCMC) generally require thousands to millions of model evaluations, and, when
coupled with an expensive model, result in excessive solve times that can render
the analysis intractable. These methods are also inherently serial, which
prohibits speedup by high performance computing. Recently, Sequential Monte 
Carlo (SMC) has emerged as a powerful alternative to MCMC. In contrast, this
method allows for parallel model evaluations, realizing significant speedup.
This software is an implementation of SMC that uses the Message Passing
Interface (MPI) to provide users general access to parallel UQ methods in
Python 2.7. The algorithm used is based on the work by Nguyen et al. ["Efficient
Sequential Monte-Carlo Samplers for Bayesian Inference" IEEE Transactions on
Signal Processing, Vol. 64, No. 5 (2016)]

To operate the code, the user supplies a computational model built in Python
2.7, defines prior distributions for each of the model parameters to be
estimated, and provides data to be used for calibration. SMC sampling can then
be conducted with ease through instantiation of the SMC class and a call to the
sample() method. The output of this process is an approximation of the parameter
posterior probability distribution conditioned on the data provided.

==========================================================================

Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.
 
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR
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
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.

