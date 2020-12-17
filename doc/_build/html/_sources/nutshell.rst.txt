.. _nutshell:

Leaspy in a nutshell
********************

Comprehensive example
---------------------

We load some synthetic data from the `leaspy.datasets` module, encapsulate them
in the main `leaspy Data container`, then we plot them with the main API `Leaspy`.

    >>> from leaspy import AlgorithmSettings, Data, Leaspy
    >>> from leaspy.datasets import Loader
    >>> alzheimer_df = Loader.load_dataset('alzheimer-multivariate')
    >>> print(alzheimer_df.columns)
    Index(['E-Cog Subject', 'E-Cog Study-partner', 'MMSE', 'RAVLT', 'FAQ',
       'FDG PET', 'Hippocampus volume ratio'],
      dtype='object')
    >>> print(alzheimer_df.head())
                          MMSE     RAVLT       FAQ   FDG PET
    ID     TIME
    GS-001 73.973183  0.111998  0.510524  0.178827  0.454605
           74.573181  0.029991  0.749223  0.181327  0.450064
           75.173180  0.121922  0.779680  0.026179  0.662006
           75.773186  0.092102  0.649391  0.156153  0.585949
           75.973183  0.203874  0.612311  0.320484  0.634809
    >>> data = Data.from_dataframe(alzheimer_df)
    >>> leaspy_logistic = Leaspy('logistic')
    >>> ax = leaspy_logistic.plotting.patient_observations(data)

    .. figure::  alzheimer-observations.png
      :align:   center

Not so engaging, right? With `leaspy`, we can derive the group average trajectory
of this population. We use the `Leaspy.fit` method by providing it the settings
for the MCMC-SAEM algorithm. Then, we plot the group average trajectory:
    >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0, progress_bar=True)
    >>> leaspy_logistic.fit(data, model_settings)
    ==> Setting seed to 0
    |##################################################|   10000/10000 iterations
    The standard deviation of the noise at the end of the calibration is:
    0.0718
    Calibration took: 5min 55s
    >>> ax2 = leaspy_logistic.plotting.average_trajectory()

    .. figure::  alzheimer-model.png
      :align:   center

We can also derive the individual trajectory of each subject. To do this,
we use the `Leaspy.personalize` method, again by providing the proper settings.
Then we plot both, the first subjects observations and trajectories:
    >>> personalize_settings = AlgorithmSettings('scipy_minimize', progress_bar=True, \
    use_jacobian=True, seed=0)
    >>> individual_parameters = leaspy_logistic.personalize(data, personalize_settings)
     ==> Setting seed to 0
    |##################################################|   200/200 subjects
    The standard deviation of the noise at the end of the personalization is:
    0.0686
    Personalization scipy_minimize took: 11s
    >>> ax = leaspy_logistic.plotting.patient_trajectories(data, individual_parameters, 'GS-001')
    >>> leaspy_logistic.plotting.patient_observations(data, 'GS-001', ax=ax)
    >>> import matplotlib.pyplot as plt
    >>> plt.legend(loc='upper left', title='Features', bbox_to_anchor=(1.05, 1))
    >>> plt.title('Subject GS-001 - observations & trajectories')

    .. figure::  alzheimer-subject_trajectories.png
      :align:   center

Using my own data
-----------------

Data format
^^^^^^^^^^^

`Leaspy` use its own data container. To use it properly, you need to provide a
`.csv` file or a `pandas.DataFrame` in the right format. Let's have a look on
the data used in the previous example:

    >>> print(alzheimer_df.head())
                          MMSE     RAVLT       FAQ   FDG PET
    ID     TIME
    GS-001 73.973183  0.111998  0.510524  0.178827  0.454605
           74.573181  0.029991  0.749223  0.181327  0.450064
           75.173180  0.121922  0.779680  0.026179  0.662006
           75.773186  0.092102  0.649391  0.156153  0.585949
           75.973183  0.203874  0.612311  0.320484  0.634809

You **MUST** have `ID` and `TIME`, either in index or in the columns. The other
columns must be the observation variables, or *features*. In this fashion,
you have one column per *feature* and one line per *visit*.

Data scale & constraints
^^^^^^^^^^^^^^^^^^^^^^^^

`Leaspy` use *linear* and *logisitic* models. The features **MUST** be increasing
with time. For the *logisitic* model, you need to rescale your data between
0 and 1.

Missing data
^^^^^^^^^^^^

`Leaspy` automatically handle missing data. However, they **MUST** be encoded
as ``numpy.nan`` in your `pandas.DataFrame`.

Going further
-------------

You can check the :ref:`user_guide` and the :ref:`full API documentation <api>`.

.. toctree::
   :numbered:
   :maxdepth: 2
