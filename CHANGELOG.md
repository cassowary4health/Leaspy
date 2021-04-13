Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

# [1.1.2] - 2021-04-13
- **Fix computation of orthonormal basis for leaspy multivariate models:**
  - **<!> this modification is breaking old multivariate models for which orthogonality**
    **between space-shifts and time-derivatie of geodesic at initial time was not respected.**
  - To prevent misusing old saved models (with betas and sources being related to the old wrong
    orthonormal basis) with new code, we added the leaspy version in the model parameters when saved,
    you'll have to use leaspy 1.0.* to run old erroneous models and use leaspy >= 1.1 to run new models
- Change of the sequence `epsilon_k` in the updates of the sufficient statistics (after burn-in phase)
  to ensure the theoretical convergence
- No use of v0_mean hyperprior for multivariate models
- Uniformize tiny MCMC std priors on population parameters
- Better initialization of velocities for multivariate models
- New method for initialization `lme` (not used by default)
- In `AlgorithmSettings`:
  - `initialization_method` is renamed `algorithm_initialization_method`
  - for fit algorithms, you can define `model_initialization_method`
- Refactorization of models attributes
- Clean-up in code and in tests

# [1.0.3] - 2021-03-03
- Fix multivariate linear model
- Fix multivariate linear & logistic_parallel jacobian computation
- Update requirements.txt and add a `__watermark__`
- Add support for new torch versions (1.2.* and >1.4 but <1.7)
- Tiny fixes on starter notebook
- Tiny fix on `Plotting`
- Clean-up in documentation

# [1.0.2] - 2021-01-05
- Jacobian for all models
- Clean univariate models
- More coherent initializations

# [1.0.1] - 2021-01-04
- First released version

