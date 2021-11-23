# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2021-12-17
- [CODE] Broad use of type annotations in Leaspy package
- [COMPAT] As a consequence support of Python 3.6 is dropped
- [COMPAT] PyTorch >=1.7 is now supported, as well as Python 3.9
- [FEAT] Custom Leaspy exceptions raised in code: cf. `leaspy.exceptions` in documentation
- [FEAT] Implementation of model _inverse_ in API to get age according to model for a given feature value: `Leaspy.estimate_ages_from_biomarker_values`
- [FIX] Simulation algorithm is fixed (shape issue with noise and bad behavior for random visits <= 0)
- [REFACT] Configuration of noise structure was reshaped:
  - models do not have a `loss` hyperparameter any more, it was replaced by `noise_model` (backward-compatibility is ensured)
  - algorithms do not have a `loss` parameter any more, it is fully controlled by noise structure of model
- [FEAT] Simulation algorithm now supports new keywords (cf. `SimulationAlgorithm`) to better control:
  - delay between simulated visits (can be random, constant or defined with a custom function)
  - number of simulated visits (possible to set min & max number of visits when random)
  - noise structure, in line with new `noise_model` implementation
- [DEFAULTS] Some default configuration values changed for some algorithms & models:
  - LME model now has `with_random_slope_age` = True by default
  - `mcmc_saem` and `scipy_minimize` now have `progress_bar` = True by default
  - `scipy_minimize` now has `use_jacobian` = True by default
  - Multivariate models now have `gaussian_diagonal` noise structure by default
  - `simulation` algorithm now has constant delay of 6 months between simulated visits by default
- [DEPRECATION] Some classes were deprecated:
  - `VisualizationToolbox` class was removed from code
  - `Plotting` class was deprecated and removed from Leaspy API
  - Some already deprecated algorithms are not supported any more (moved in `algo/_legacy` folder)
- [BROWSER] Browser web-app was improved
- [LICENSE] Changing GPL license to BSD 3-Clause license - active only for current and future releases
- [REFACT] Readers now implement more checks on input data; CSVDataReader now calls `pandas.read_csv` internally
- [TESTS] Refactoring of tests with a new `LeaspyTestCase`, have a look at `tests/README.md` if you want to add or modify them

## [1.1.2] - 2021-04-13
- **Fix computation of orthonormal basis for leaspy multivariate models:**
  - **<!> this modification is breaking old multivariate models for which orthogonality**
    **between space-shifts and time-derivative of geodesic at initial time was not respected.**
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

## [1.0.3] - 2021-03-03
- Fix multivariate linear model
- Fix multivariate linear & logistic_parallel jacobian computation
- Update requirements.txt and add a `__watermark__`
- Add support for new torch versions (1.2.* and >1.4 but <1.7)
- Tiny fixes on starter notebook
- Tiny fix on `Plotting`
- Clean-up in documentation

## [1.0.2] - 2021-01-05
- Jacobian for all models
- Clean univariate models
- More coherent initializations

## [1.0.1] - 2021-01-04
- First released version

