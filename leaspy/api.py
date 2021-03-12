import pandas as pd

from leaspy.algo.algo_factory import AlgoFactory
from leaspy.io.data.dataset import Dataset
from leaspy.io.logs.visualization.plotting import Plotting
from leaspy.io.settings.model_settings import ModelSettings
from leaspy.models.model_factory import ModelFactory


class Leaspy:
    r"""
    Main API used to fit models, run algorithms and simulations.
    This is the main class of the Leaspy package.

    Parameters
    ----------
    model_name: str
        The name of the model that will be used for the computations.
        The available models are:
            * ``'logistic'`` - suppose that every modality follow a logistic curve across time. This model performs a dimensionality reduction of the modalities.
            * ``'logistic_parallel'`` - idem & suppose also that every modality have the same slope at inflexion point
            * ``'linear'`` - suppose that every modality follow a linear curve across time. This model performs a dimensionality reduction of the modalities.
            * ``'univariate_logisitic'`` - a 'logistic' model for a single modality => do not perform a dimensionality reduction.
            * ``'univariate_linear'`` - idem with a 'linear' model.
            * ``'constant'`` - benchmark model for constant predictions.
            * ``'lme'`` - benchmark model for classical linear mixed-effects model.

    **kwargs:
        Keyword arguments directly passed to the model for its initialization (through :meth:`.ModelFactory.model`).
        Refer to the corresponding model to know possible arguments.

        source_dimension: int, optional
            `For multivariate models only`.
            Set the spatial variability degree of freedom.
            This number MUST BE lower than the number of features.
            By default, this number is equal to square root of the number of features.

    Attributes
    ----------
    model : :class:`~.models.abstract_model.AbstractModel`
        Model used for computations, is an instance of `AbstractModel`.
    type : str (read-only)
        Name of the model - will be one of the names listed above.
    plotting : :class:`~.io.logs.visualization.plotting.Plotting`
        Main class for visualization.

    See also
    --------
    leaspy.models
    """

    def __init__(self, model_name, **kwargs):
        """
        Instantiate a Leaspy class object.
        """
        self.model = ModelFactory.model(model_name, **kwargs)
        #self.type = model_name # is a property instead
        self.plotting = Plotting(self.model)

    @property
    def type(self) -> str:
        return self.model.name

    def fit(self, data, algorithm_settings):
        r"""
        Estimate the model's parameters :math:`\theta` for a given dataset and a given algorithm.
        These model's parameters correspond to the fixed-effects of the mixed-effects model.

        Parameters
        ----------
        data : :class:`.Data`
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        algorithm_settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.

        Examples
        --------
        Fit a logistic model on a longitudinal dataset, display the group parameters and plot the
        group average trajectory.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> from leaspy.datasets import Loader
        >>> putamen_df = Loader.load_dataset('parkinson-putamen')
        >>> data = Data.from_dataframe(putamen_df)
        >>> leaspy_logistic = Leaspy('univariate_logistic')
        >>> settings = AlgorithmSettings('mcmc_saem', progress_bar=True, seed=0)
        >>> leaspy_logistic.fit(data, settings)
         ==> Setting seed to 0
        |##################################################|   10000/10000 iterations
        The standard deviation of the noise at the end of the calibration is:
        0.0213
        Calibration took: 30s
        >>> print(str(leaspy_logistic.model))
        === MODEL ===
        g : tensor([-1.1744])
        tau_mean : 68.56787872314453
        tau_std : 10.12782096862793
        xi_mean : -2.3396952152252197
        xi_std : 0.5421289801597595
        noise_std : 0.021265486255288124
        >>> leaspy_logistic.plotting.average_trajectory()
        """
        algorithm = AlgoFactory.algo("fit", algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(self.model, dataset)

        # Update plotting
        self.plotting.update_model(self.model)

    def calibrate(self, data, algorithm_settings):
        r"""
        Duplicates of the :meth:`~.Leaspy.fit` method.
        """
        self.fit(data, algorithm_settings)

    def personalize(self, data, settings, return_noise=False):
        r"""
        From a model, estimate individual parameters for each `ID` of a given dataset.
        These individual parameters correspond to the random-effects :math:`(z_{i,j})` of the mixed-effects model.

        Parameters
        ----------
        data : :class:`.Data`
            Contains the information of the individuals, in particular the time-points :math:`(t_{i,j})` and the observations :math:`(y_{i,j})`.
        settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.
        return_noise: bool (default False)
            Returns a tuple (individual_parameters, noise_std) if True

        Returns
        -------
        ips : :class:`.IndividualParameters`
            Contains individual parameters

        if return_noise is True:
            ips : :class:`.IndividualParameters`
            noise_std : :class:`torch.Tensor`

        Examples
        --------
        Compute the individual parameters for a given longitudinal dataset and calibrated model, then
        display the histogram of the log-acceleration:

        >>> from leaspy import AlgorithmSettings, Data
        >>> from leaspy.datasets import Loader
        >>> leaspy_logistic = Loader.load_leaspy_instance('parkinson-putamen-train')
        >>> putamen_df = Loader.load_dataset('parkinson-putamen')
        >>> data = Data.from_dataframe(putamen_df)
        >>> personalize_settings = AlgorithmSettings('scipy_minimize', progress_bar=True, use_jacobian=True, seed=0)
        >>> individual_parameters = leaspy_logistic.personalize(data, personalize_settings)
         ==> Setting seed to 0
        |##################################################|   200/200 subjects
        The standard deviation of the noise at the end of the personalization is:
        0.0191
        Personalization scipy_minimize took: 5s
        >>> ip_df = individual_parameters.to_dataframe()
        >>> ip_df[['xi']].hist()
        """
        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo("personalize", settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        individual_parameters, noise_std = algorithm.run(self.model, dataset)

        if return_noise:
            return individual_parameters, noise_std
        else:  # default
            return individual_parameters

    def estimate(self, timepoints, individual_parameters, *, to_dataframe=None):
        r"""
        Return the model values for individuals characterized by their individual parameters :math:`z_i` at time-points :math:`(t_{i,j})_j`

        Parameters
        ----------
        timepoints : dictionary {string/int: array_like[numeric]} or :class:`pandas.MultiIndex`
            Contains, for each individual, the time-points to estimate.
            It can be a unique time-point or a list of time-points.
        individual_parameters : :class:`.IndividualParameters`
            Corresponds to the individual parameters of individuals.
        to_dataframe : bool or None (default)
            Whether to output a dataframe of estimations?
            If None: default is to be True if and only if timepoints is a `pandas.MultiIndex`

        Returns
        -------
        individual_trajectory : dict or :class:`pandas.DataFrame` (depending on `to_dataframe` flag)
            Key: patient indices.
            Value: :class:`numpy.ndarray` of the estimated value, in the shape
            (number of timepoints, number of features)

        Examples
        --------
        Given the individual parameters of two subjects, estimate the features of the first
        at 70, 74 and 80 years old and at 71 and 72 years old for the second.

        >>> from leaspy.datasets import Loader
        >>> leaspy_logistic = Loader.load_leaspy_instance('parkinson-putamen-train')
        >>> individual_parameters = Loader.load_individual_parameters('parkinson-putamen-train')
        >>> timepoints = {'GS-001': (70, 74, 80), 'GS-002': (71, 72)}
        >>> estimations = leaspy_logistic.estimate(timepoints, individual_parameters)
        """
        estimations = {}

        ix = None
        # get timepoints to estimate from index
        if isinstance(timepoints, pd.MultiIndex):

            # default output is pd.DataFrame when input as pd.MultiIndex
            if to_dataframe is None:
                to_dataframe = True

            ix = timepoints # keep for future
            timepoints = {pat_id: ages.values for pat_id, ages in timepoints.to_frame()['TIME'].groupby('ID')}

        for index, time in timepoints.items():
            ip = individual_parameters[index]
            est = self.model.compute_individual_trajectory(time, ip)
            estimations[index] = est[0].numpy() # 1 individual at a time (first dimension of tensor)

        # convert to proper dataframe
        if to_dataframe:
            estimations = pd.concat({
                pat_id: pd.DataFrame(ests, index=timepoints[pat_id], columns=self.model.features)
                for pat_id, ests in estimations.items()
            }, names=['ID','TIME'])

            # reindex back to given index being careful to index order (join so to handle multi-levels cases)
            if ix is not None:
                estimations = pd.DataFrame([], index=ix).join(estimations)

        return estimations

    def simulate(self, individual_parameters, data, settings):
        r"""
        Generate longitudinal synthetic patients data from a given model, a given collection of individual parameters
        and some given settings.

        This procedure learn the joined distribution of the individual parameters and baseline age of the subjects
        present in ``individual_parameters`` and ``data`` respectively to sample new patients from this joined distribution.
        The model is used to compute for each patient their scores from the individual parameters.
        The number of visits per patients is set in ``settings['parameters']['mean_number_of_visits']`` and
        ``settings['parameters']['std_number_of_visits']`` which are set by default to 6 and 3 respectively.

        Parameters
        ----------
        individual_parameters : :class:`.IndividualParameters`
            Contains the individual parameters.
        data : :class:`.Data`
            Data object
        settings : :class:`.AlgorithmSettings`
            Contains the algorithm's settings.

        Returns
        -------
        simulated_data : :class:`~.io.outputs.result.Result`
            Contains the generated individual parameters & the corresponding generated scores.

        Notes
        -----
        To generate a new subject, first we estimate the joined distribution of the individual parameters and the
        reparametrized baseline ages. Then, we randomly pick a new point from this distribution, which define the
        individual parameters & baseline age of our new subjects. Then, we generate the timepoints
        following the baseline age. Then, from the model and the generated timepoints and individual parameters, we
        compute the corresponding values estimations. Then, we add some gaussian noise to these estimations. The level
        of noise is, by default, equal to the corresponding ``'noise_std'`` parameter of the model. You can choose
        to set your own noise value.

        Examples
        --------
        Use a calibrated model & individual parameters to simulate new subjects similar to the ones you have:

        >>> from leaspy import AlgorithmSettings, Data
        >>> from leaspy.datasets import Loader
        >>> putamen_df = Loader.load_dataset('parkinson-putamen-train_and_test')
        >>> data = Data.from_dataframe(putamen_df.xs('train', level='SPLIT'))
        >>> leaspy_logistic = Loader.load_leaspy_instance('parkinson-putamen-train')
        >>> individual_parameters = Loader.load_individual_parameters('parkinson-putamen-train')
        >>> simulation_settings = AlgorithmSettings('simulation', seed=0)
        >>> simulated_data = leaspy_logistic.simulate(individual_parameters, data, simulation_settings)
         ==> Setting seed to 0
        >>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).head())
                                                  PUTAMEN
        ID                    TIME
        Generated_subject_001 63.611107  0.556399
                              64.111107  0.571381
                              64.611107  0.586279
                              65.611107  0.615718
                              66.611107  0.644518
        >>> print(simulated_data.get_dataframe_individual_parameters().tail())
                                     tau        xi
        ID
        Generated_subject_096  46.771028 -2.483644
        Generated_subject_097  73.189964 -2.513465
        Generated_subject_098  57.874967 -2.175362
        Generated_subject_099  54.889400 -2.069300
        Generated_subject_100  50.046972 -2.259841

        By default, you have simulate 100 subjects, with an average number of visit at 6 & and standard deviation
        is the number of visits equal to 3. Let's say you want to simulate 200 subjects, everyone of them having
        ten visits exactly:

        >>> simulation_settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=200, \
        mean_number_of_visits=10, std_number_of_visits=0)
         ==> Setting seed to 0
        >>> simulated_data = leaspy_logistic.simulate(individual_parameters, data, simulation_settings)
        >>> print(simulated_data.data.to_dataframe().set_index(['ID', 'TIME']).tail())
                                          PUTAMEN
        ID                    TIME
        Generated_subject_200 72.119949  0.829185
                              73.119949  0.842113
                              74.119949  0.854271
                              75.119949  0.865680
                              76.119949  0.876363

        By default, the generated subjects are named `'Generated_subject_001'`, `'Generated_subject_002'` and so on.
        Let's say you want a shorter name, for exemple `'GS-001'`. Furthermore, you want to set the level of noise
        arround the subject trajectory when generating the observations:

        >>> simulation_settings = AlgorithmSettings('simulation', seed=0, prefix='GS-', noise=.2)
        >>> simulated_data = leaspy_logistic.simulate(individual_parameters, data, simulation_settings)
         ==> Setting seed to 0
        >>> print(simulated_data.get_dataframe_individual_parameters().tail())
                      tau        xi
        ID
        GS-096  46.771028 -2.483644
        GS-097  73.189964 -2.513465
        GS-098  57.874967 -2.175362
        GS-099  54.889400 -2.069300
        GS-100  50.046972 -2.259841
        """
        # Check if model has been initialized
        self.check_if_initialized()

        algorithm = AlgoFactory.algo("simulate", settings)
        simulated_data = algorithm.run(self.model, individual_parameters, data)
        return simulated_data

    @classmethod
    def load(cls, path_to_model_settings):
        r"""
        Instantiate a Leaspy object from json model parameter file or the corresponding dictionary

        This function can be used to load a pre-trained model.

        Parameters
        ----------
        path_to_model_settings: str or dict
            Path to the model's settings json file or dictionary of model parameters

        Returns
        -------
        :class:`.Leaspy`
            An instanced Leaspy object with the given population parameters :math:`\theta`.

        Examples
        --------
        Load a univariate logistic pre-trained model.

        >>> from leaspy import Leaspy
        >>> from leaspy.datasets.loader import model_paths
        >>> leaspy_logistic = Leaspy.load(model_paths['parkinson-putamen-train'])
        >>> print(str(leaspy_logistic.model))
        === MODEL ===
        g : tensor([-0.7901])
        tau_mean : 64.18125915527344
        tau_std : 10.199116706848145
        xi_mean : -2.346343994140625
        xi_std : 0.5663877129554749
        noise_std : 0.021229960024356842
        """
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)

        # dirty... logic should be changed to be compatible with models without MCMC toolbox (constant model or LME model)
        if hasattr(leaspy.model, 'initialize_MCMC_toolbox'):
            leaspy.model.initialize_MCMC_toolbox()

        leaspy.model.is_initialized = True

        # Update plotting
        leaspy.plotting.update_model(leaspy.model)

        return leaspy

    def save(self, path, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.

        Examples
        --------
        Load the univariate dataset ``'parkinson-putamen'``, calibrate the model & save it:

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> from leaspy.datasets import Loader
        >>> putamen_df = Loader.load_dataset('parkinson-putamen')
        >>> data = Data.from_dataframe(putamen_df)
        >>> leaspy_logistic = Leaspy('univariate_logistic')
        >>> settings = AlgorithmSettings('mcmc_saem', progress_bar=True, seed=0)
        >>> leaspy_logistic.fit(data, settings)
         ==> Setting seed to 0
        |##################################################|   10000/10000 iterations
        The standard deviation of the noise at the end of the calibration is:
        0.0213
        Calibration took: 30s
        >>> leaspy_logistic.save('leaspy-logistic-model_parameters-seed0.json', indent=2)
        """
        self.check_if_initialized()
        self.model.save(path, **kwargs)

    def check_if_initialized(self):
        """
        Check if model is initialized.

        Raises
        ------
        ValueError
            Raise an error if the model has not been initialized.
        """
        if not self.model.is_initialized:
            raise ValueError("Model has not been initialized")
