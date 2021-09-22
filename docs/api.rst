.. _api:

=================
API Documentation
=================

Full API documentation of the *Leaspy* Python package.


:mod:`leaspy.api`: **Main API**
===============================
The main class, from which you can instantiate and calibrate a model, personalize it to
a given set a subjects, estimate trajectories and simulate synthetic data.

.. automodule:: leaspy.api

.. currentmodule:: leaspy.api

.. autosummary::
   :toctree: generated/
   :template: class.rst

   Leaspy

:mod:`leaspy.models`: **Models**
================================
Available models in `Leaspy`.

.. currentmodule:: leaspy.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_factory.ModelFactory
    abstract_model.AbstractModel
    univariate_model.UnivariateModel
    abstract_multivariate_model.AbstractMultivariateModel
    multivariate_model.MultivariateModel
    multivariate_parallel_model.MultivariateParallelModel
    .. generic_model.GenericModel
    lme_model.LMEModel
    constant_model.ConstantModel

:mod:`leaspy.models.utils.attributes`: **Models' attributes**
-------------------------------------------------------------
Attributes used by the models.

.. currentmodule:: leaspy.models.utils.attributes

.. autosummary::
    :toctree: generated/
    :template: class.rst

    attributes_factory.AttributesFactory
    abstract_attributes.AbstractAttributes
    abstract_manifold_model_attributes.AbstractManifoldModelAttributes
    linear_attributes.LinearAttributes
    logistic_attributes.LogisticAttributes
    logistic_parallel_attributes.LogisticParallelAttributes

:mod:`leaspy.models.utils.initialization`: **Initialization methods**
---------------------------------------------------------------------
Available methods to initialize model parameters before a fit.

.. currentmodule:: leaspy.models.utils.initialization

.. autosummary::
  :toctree: generated/
  :template: function.rst

  model_initialization.initialize_parameters
  .. model_initialization.initialize_linear
  .. model_initialization.initialize_logistic
  .. model_initialization.initialize_logistic_parallel

:mod:`leaspy.algo`: **Algorithms**
==================================
Contains all algorithms used in the package.

.. automodule:: leaspy.algo
   :no-members:
   :no-inherited-members:

.. currentmodule:: leaspy.algo

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_algo.AbstractAlgo
    algo_factory.AlgoFactory

:mod:`leaspy.algo.fit`: **Fit algorithms**
------------------------------------------
Algorithms used to calibrate (fit) a model.

.. currentmodule:: leaspy.algo.fit

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_fit_algo.AbstractFitAlgo
    abstract_mcmc.AbstractFitMCMC
    tensor_mcmcsaem.TensorMCMCSAEM

:mod:`leaspy.algo.personalize`: **Personalization algorithms**
--------------------------------------------------------------
Algorithms used to personalize a model to given subjects.

.. currentmodule:: leaspy.algo.personalize

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_personalize_algo.AbstractPersonalizeAlgo
    scipy_minimize.ScipyMinimize

:mod:`leaspy.algo.simulate`: **Simulation algorithms**
------------------------------------------------------
Algorithm to simulate synthetic observations and individual parameters.

.. currentmodule:: leaspy.algo.simulate

.. autosummary::
    :toctree: generated/
    :template: class.rst

    simulate.SimulationAlgorithm

:mod:`leaspy.algo.others`: **Other algorithms**
-----------------------------------------------
Reference algorithms to use with reference models (for benchmarks).

.. currentmodule:: leaspy.algo.others

.. autosummary::
    :toctree: generated/
    :template: class.rst

    constant_prediction_algo.ConstantPredictionAlgorithm
    lme_fit.LMEFitAlgorithm
    lme_personalize.LMEPersonalizeAlgorithm

:mod:`leaspy.algo.samplers`: **Samplers**
-----------------------------------------
Samplers used by the MCMC algorithms.

.. currentmodule:: leaspy.algo.samplers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_sampler.AbstractSampler
    gibbs_sampler.GibbsSampler

:mod:`leaspy.dataset`: **Datasets**
===================================
Give access to some synthetic longitudinal observations mimicing cohort of subjects with neurodegenerative disorders,
as well as calibrated models and computed individual parameters.

.. currentmodule:: leaspy.datasets

.. autosummary::
    :toctree: generated/
    :template: class.rst

    loader.Loader

:mod:`leaspy.io`: **Inputs / Outputs**
======================================
Containers classes used as input / ouputs in the `Leaspy` package.

:mod:`leaspy.io.data`: **Data containers**
------------------------------------------

.. currentmodule:: leaspy.io.data

.. autosummary::
    :toctree: generated/
    :template: class.rst

    data.Data
    dataset.Dataset
    .. individual_data.IndividualData
    .. csv_data_reader.CSVDataReader
    .. dataframe_data_reader.DataframeDataReader

:mod:`leaspy.io.settings`: **Settings classes**
-----------------------------------------------

.. currentmodule:: leaspy.io.settings

.. autosummary::
    :toctree: generated/
    :template: class.rst

    model_settings.ModelSettings
    algorithm_settings.AlgorithmSettings
    outputs_settings.OutputsSettings

:mod:`leaspy.io.outputs`: **Outputs classes**
---------------------------------------------

.. currentmodule:: leaspy.io.outputs

.. autosummary::
    :toctree: generated/
    :template: class.rst

    individual_parameters.IndividualParameters
    .. result.Result

:mod:`leaspy.io.realizations`: **Realizations classes**
-------------------------------------------------------
Internal classes used for random variables in MCMC algorithms.

.. currentmodule:: leaspy.io.realizations

.. autosummary::
    :toctree: generated/
    :template: class.rst

    realization.Realization
    collection_realization.CollectionRealization

:mod:`leaspy.exceptions`: **Exceptions**
========================================
Custom `Leaspy` exceptions raised in code.

.. currentmodule:: leaspy.exceptions

.. autosummary::
    :toctree: generated/
    :template: exception_class.rst

    LeaspyException
    LeaspyTypeError
    LeaspyInputError
    LeaspyDataInputError
    LeaspyModelInputError
    LeaspyAlgoInputError
    LeaspyIndividualParamsInputError
