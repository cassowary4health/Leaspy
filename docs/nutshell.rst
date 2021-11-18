.. _nutshell:

Leaspy in a nutshell
********************

Comprehensive example
---------------------

We first load synthetic data from the `leaspy.datasets` to get of a grasp of longitudinal data.


    >>> from leaspy import AlgorithmSettings, Data, Leaspy
    >>> from leaspy.datasets import Loader
    >>> alzheimer_df = Loader.load_dataset('alzheimer-multivariate')
    >>> print(alzheimer_df.columns)
    Index(['E-Cog Subject', 'E-Cog Study-partner', 'MMSE', 'RAVLT', 'FAQ',
       'FDG PET', 'Hippocampus volume ratio'],
      dtype='object')
    >>> alzheimer_df = alzheimer_df[['MMSE', 'RAVLT', 'FAQ', 'FDG PET']]
    >>> print(alzheimer_df.head())
                          MMSE     RAVLT       FAQ   FDG PET
    ID     TIME
    GS-001 73.973183  0.111998  0.510524  0.178827  0.454605
           74.573181  0.029991  0.749223  0.181327  0.450064
           75.173180  0.121922  0.779680  0.026179  0.662006
           75.773186  0.092102  0.649391  0.156153  0.585949
           75.973183  0.203874  0.612311  0.320484  0.634809

The data corresponds to repeated visits (`TIME` index) of different participants (`ID` index).
Each visit corresponds to the measurement of 4 different variables : the MMSE, the RAVLT, the FAQ and the FDG PET.

If plotted, the data would look like the following :

    .. figure::  _static/images/alzheimer-observations.png
      :align:   center

where each color corresponds to a variable, and the connected dots corresponds
to the repeated visits of a single participant.

Not very engaging, right ? To go a step further, let's first encapsulate the data into the main `leaspy Data container`.


    >>> data = Data.from_dataframe(alzheimer_df)

Leaspy core functionality is to estimate the group-average trajectory
of the different variables that are measured in a population. Let's initialize the leaspy object

    >>> leaspy_logistic = Leaspy('logistic')

as well as the Algorithms needed to estimate the group-average trajectory :

    >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0, progress_bar=True)

We then use the `Leaspy.fit` method to estimate the group average trajectory:


    >>> leaspy_logistic.fit(data, model_settings)
    ==> Setting seed to 0
    |##################################################|   10000/10000 iterations
    The standard deviation of the noise at the end of the calibration is:
    0.0718
    Calibration took: 5min 55s

If we were to plot the measured average progression of the variables
- see `started example notebook <https://gitlab.com/icm-institute/aramislab/leaspy>`_ for details -
it would look like the following

    .. figure::  _static/images/alzheimer-model.png
      :align:   center

We can also derive the individual trajectory of each subject. To do this,
we use the `Leaspy.personalize` method, again by providing the proper settings.

    >>> personalize_settings = AlgorithmSettings('scipy_minimize', progress_bar=True, \
    use_jacobian=True, seed=0)
    >>> individual_parameters = leaspy_logistic.personalize(data, personalize_settings)
     ==> Setting seed to 0
    |##################################################|   200/200 subjects
    The standard deviation of the noise at the end of the personalization is:
    0.0686
    Personalization scipy_minimize took: 11s

Plotting the input participant data agains its personalization would give the following
- see `started example notebook <https://gitlab.com/icm-institute/aramislab/leaspy>`_ for details.

    .. figure::  _static/images/alzheimer-subject_trajectories.png
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
You can also dive into the `started example <https://gitlab.com/icm-institute/aramislab/leaspy>`_ of the Leaspy
repository.
The `Disease Progression Modelling <https://disease-progression-modelling.github.io/>`_ website also hosts
a `mathematical introduction <https://disease-progression-modelling.github.io/pages/models/disease_course_mapping.html>`_
and `tutorials <https://disease-progression-modelling.github.io/pages/notebooks/disease_course_mapping/disease_course_mapping.html>`_
to Leaspy.
