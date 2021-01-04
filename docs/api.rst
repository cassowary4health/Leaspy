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
Algorithms used to calibrate a model.

.. currentmodule:: leaspy.algo.fit

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_fit_algo.AbstractFitAlgo
    abstract_mcmc.AbstractFitMCMC
    tensor_mcmcsaem.TensorMCMCSAEM

:mod:`leaspy.algo.fit`: **Personalization algorithms**
------------------------------------------------------
Algorithms used to personalize a model to a given set of subjects.

.. currentmodule:: leaspy.algo.personalize

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_personalize_algo.AbstractPersonalizeAlgo
    scipy_minimize.ScipyMinimize

:mod:`leaspy.algo.fit`: **Samplers**
------------------------------------
Samplers used by the algorithms.

.. currentmodule:: leaspy.algo.samplers

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_sampler.AbstractSampler
    gibbs_sampler.GibbsSampler
    hmc_sampler.HMCSampler

:mod:`leaspy.algo.fit`: **Simulation algorithms**
--------------------------------------------------
Algorithm to simulate synthetic observations and individual parameters.

.. currentmodule:: leaspy.algo.simulate

.. autosummary::
    :toctree: generated/
    :template: class.rst

    simulate.SimulationAlgorithm

:mod:`leaspy.dataset`: **Datasets**
==========================================
Give access to some synthetic longitudinal observations mimicking cohort of subjects with neurodegenerative disorders,
as well as calibrated models and computed individual parameters.

.. currentmodule:: leaspy.datasets

.. autosummary::
    :toctree: generated/
    :template: class.rst

    loader.Loader

:mod:`leaspy.io`: **Inputs / Outputs**
======================================
Containers class objects used as input / ouputs in the `Leaspy` package.

:mod:`leaspy.io.data`: **Data containers**
------------------------------------------

.. currentmodule:: leaspy.io.data

.. autosummary::
    :toctree: generated/
    :template: class.rst

    csv_data_reader.CSVDataReader
    data.Data
    dataframe_data_reader.DataframeDataReader
    dataset.Dataset
    individual_data.IndividualData

:mod:`leaspy.io.outputs`: **Outputs class objects**
---------------------------------------------------

.. currentmodule:: leaspy.io.outputs

.. autosummary::
    :toctree: generated/
    :template: class.rst

    individual_parameters.IndividualParameters

:mod:`leaspy.io.outputs`: **Realizations class objects**
--------------------------------------------------------

.. currentmodule:: leaspy.io.realizations

.. autosummary::
    :toctree: generated/
    :template: class.rst

    collection_realization.CollectionRealization
    realization.Realization

:mod:`leaspy.io.settings`: **Settings class objects**
-----------------------------------------------------

.. currentmodule:: leaspy.io.settings

.. autosummary::
    :toctree: generated/
    :template: class.rst

    algorithm_settings.AlgorithmSettings
    model_settings.ModelSettings
    outputs_settings.OutputsSettings

:mod:`leaspy.models`: **Models**
================================
Available models in `Leaspy`.

.. currentmodule:: leaspy.models

.. autosummary::
    :toctree: generated/
    :template: class.rst

    abstract_model.AbstractModel
    abstract_multivariate_model.AbstractMultivariateModel
    model_factory.ModelFactory
    multivariate_model.MultivariateModel
    multivariate_parallel_model.MultivariateParallelModel
    univariate_model.UnivariateModel

:mod:`leaspy.models.utils.attributes`: **Models' attributes**
-------------------------------------------------------------
Attributes used by the models.

.. currentmodule:: leaspy.models.utils.attributes

.. autosummary::
    :toctree: generated/
    :template: class.rst

    attributes_abstract.AttributesAbstract
    attributes_factory.AttributesFactory
    attributes_linear.AttributesLinear
    attributes_logistic.AttributesLogistic
    attributes_logistic_parallel.AttributesLogisticParallel
    attributes_univariate.AttributesUnivariate

:mod:`leaspy.models.utils.model_initialization`: **Initialization methods**
---------------------------------------------------------------------------
.. currentmodule:: leaspy.models.utils.model_initialization

.. autosummary::
  :toctree: generated/
  :template: function.rst

  model_initialization.compute_patient_slopes_distribution
  model_initialization.compute_patient_time_distribution
  model_initialization.compute_patient_values_distribution
  model_initialization.initialize_linear
  model_initialization.initialize_logistic
  model_initialization.initialize_logistic_parallel
  model_initialization.initialize_parameters
