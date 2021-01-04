import json
import os
import warnings

from leaspy.io.settings import default_data_dir
from leaspy.io.settings.outputs_settings import OutputsSettings


class AlgorithmSettings:
    """
    Used to set the algorithms' settings.
    All parameters, except the choice of the algorithm, is set by default.
    The user can overwrite all default settings.

    Parameters
    ----------
    name: {'mcmc_saem', 'scipy_minimize', 'simulation', 'mean_real', 'gradient_descent_personalize', 'mode_real'}
        The algorithm's name.
    n_iter: int, optional
        Number of iteration. There is no stopping criteria for the all the MCMC SAEM algorithms.
    n_burn_in_iter: int, optional
        Number of iteration during burning phase, used for the MCMC SAEM algorithms.
    seed: int, optional, default None
        Used for stochastic algorithms.
    use_jacobian: bool, optional, default False
        Used in ``scipy_minimize`` algorithm to perform a `LBFGS` instead of a `Powell` algorithm.
    n_jobs: int, optional, default 1
        Used in ``scipy_minimize`` algorithm to accelerate calculation with parallel derivation using joblib.
    loss: {'MSE', 'MSE_diag_noise', 'crossentropy'}, optional, default 'MSE'
        The wanted loss.
            * MSE: MSE of all features
            * MSE_diag_noise: MSE per feature
            * crossentropy: used when the features are binary
    progress_bar: bool, optional, default False
        Used to display a progress bar during computation.

    Attributes
    ----------
    name: {'mcmc_saem', 'scipy_minimize', 'simulation', 'mean_real', 'gradient_descent_personalize', 'mode_real'}
        The algorithm's name.
    **kwargs:
        * seed: int, optional, default None
            Used for stochastic algorithms.
        * loss: {'MSE', 'MSE_diag_noise', 'crossentropy'}, optional, default 'MSE'
            The wanted loss.
                * MSE: MSE of all features
                * MSE_diag_noise: MSE per feature
                * crossentropy: used when the features are binary
        * parameters: dict
            Contains the other parameters: `n_iter`, `n_burn_in_iter`, `use_jacobian`, `n_jobs` & `progress_bar`.
        * logs: NoneType or leaspy.io.settings.outputs_settings.OutputsSettings
            Used to create a ``logs`` file during a model calibration containing convergence information.

    Methods
    -------
    load(path_to_algorithm_settings)
        Instantiate a AlgorithmSettings object from a json file.
    save(self, path, **kwargs):
        Save an AlgorithmSettings object in a json file.
    set_logs(path, **kwargs):
        Use this method to monitor the convergence of a model callibration. It create graphs and csv files of the
        values of the population parameters (fixed effects) during the callibration
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = None
        self.seed = None
        self.initialization_method = None
        self.loss = None
        self.logs = None

        if name in ['mcmc_saem', 'scipy_minimize', 'simulation', 'mean_real', 'gradient_descent_personalize',
                    'mode_real']:
            self._load_default_values(os.path.join(default_data_dir, 'default_' + name + '.json'))
        else:
            raise ValueError('The algorithm name >>>{0}<<< you provided does not exist'.format(name))
        self._manage_kwargs(kwargs)

    @classmethod
    def load(cls, path_to_algorithm_settings):
        """
        Instantiate a AlgorithmSettings object a from json file.

        Parameters
        ----------
        path_to_algorithm_settings: str
            Path of the json file.

        Returns
        -------
        leaspy.io.settings.algorithm_settings.AlgorithmSettings
            An instanced of AlgorithmSettings with specified parameters.

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
        >>> leaspy_univariate = AlgorithmSettings.load('outputs/leaspy-univariate_model-settings.json')
        """
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        if 'name' not in settings.keys():
            raise ValueError("Your json file must contain a 'name' attribute!")
        algorithm_settings = cls(settings['name'])

        if 'parameters' in settings.keys():
            print("You overwrote the algorithm default parameters")
            algorithm_settings.parameters = cls._get_parameters(settings)

        if 'seed' in settings.keys():
            print("You overwrote the algorithm default seed")
            algorithm_settings.seed = cls._get_seed(settings)

        if 'initialization_method' in settings.keys():
            print("You overwrote the algorithm default initialization method")
            algorithm_settings.initialization_method = cls._get_initialization_method(settings)

        if 'loss' in settings.keys():
            print('You overwrote the algorithm default loss')
        algorithm_settings.loss = cls._get_loss(settings)

        return algorithm_settings

    def save(self, path, **kwargs):
        """
        Save an AlgorithmSettings object in a json file.

        Parameters
        ----------
        path: str
            Path to store the AlgorithmSettings.
        **kwargs
            Keyword arguments for json.dump method.

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
        >>> settings = AlgorithmSettings('scipy_minimize', seed=42, loss='MSE_diag_noise', n_jobs=-1, use_jacobian=True, progress_bar=True)
        >>> settings.save('outputs/scipy_minimize-settings.json', indent=2)
        """
        json_settings = {
            "name": self.name,
            "seed": self.seed,
            "parameters": self.parameters,
            "initialization_method": self.initialization_method,
            "loss": self.loss,
            "logs": self.logs
        }

        with open(os.path.join(path), "w") as json_file:
            json.dump(json_settings, json_file, **kwargs)

    def set_logs(self, path, **kwargs):
        """
        Use this method to monitor the convergence of a model callibration. It create graphs and csv files of the
        values of the population parameters (fixed effects) during the callibration

        Parameters
        ----------
        path: str
            The path of the folder to store the graphs and csv files.
        **kwargs:
            * console_print_periodicity: int or NoneType, optional, default 50
                Display logs in the console/terminal every N iterations.
            * plot_periodicity: int or NoneType, optional, default 100
                Saves the values to display in pdf every N iterations.
            * save_periodicity: int or NoneType, optional, default 50
                Saves the values in csv files every N iterations.
            * overwrite_logs_folder: bool, optionl, default False
                Set it to ``True`` to overwrite the content of the folder in ``path``.

        Notes
        -----
        By default, if the folder given in ``path`` allready exists, the method will raise an error.
        To overwrite the content of the folder, set ``overwrite_logs_folder`` it to ``True``.

        Raises
        ------
        ValueError
            If the folder given in ``path`` allready exists and if ``overwrite_logs_folder`` is set to ``False``.
        """
        settings = {
            'path': path,
            'console_print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50,
            'overwrite_logs_folder': False
        }

        for k, v in kwargs.items():
            if k in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']:
                if (type(v) != int) and (v is not None):
                    raise TypeError(f'You must provide a integer to the input <{k}>! '
                                    f'You provide {v} of type {type(v)}.')
                settings[k] = v
            elif k in ['overwrite_logs_folder']:
                if type(v) != bool:
                    raise TypeError(f'You must provide a boolean to the input <{k}>! '
                                    f'You provide {v} of type {type(v)}.')
                settings[k] = v
            else:
                warnings.warn("The kwargs {} you provided is not correct".format(k))

        self.logs = OutputsSettings(settings)

    def _manage_kwargs(self, kwargs):
        if 'seed' in kwargs.keys():
            self.seed = self._get_seed(kwargs)
        if 'initialization_method' in kwargs.keys():
            self.initialization_method = self._get_initialization_method(kwargs)
        if 'loss' in kwargs.keys():
            self.loss = self._get_loss(kwargs)

        for k, v in kwargs.items():
            if k in ['seed', 'initialization_method', 'loss']:
                continue

            if k in self.parameters.keys():
                self.parameters[k] = v
            else:
                warning_message = "The parameter key : >>>{0}<<< you provided is unknown".format(k)
                warnings.warn(warning_message)

        if self.name == 'mcmc_saem':
            if 'n_iter' in kwargs.keys() and ('n_burn_in_iter' not in kwargs.keys() or kwargs['n_burn_in_iter'] is None):
                self.parameters['n_burn_in_iter'] = int(0.9 * kwargs['n_iter'])

            # TODO : For Raphael : what does it mean? Because there are already default value.
            # TODO : Thus, either default value for annealing/iter in the json, either here. Not both.
            if 'n_iter' in kwargs.keys() and 'annealing' not in kwargs.keys():
                self.parameters['annealing']["n_iter"] = int(0.5 * kwargs['n_iter'])

    def _load_default_values(self, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        AlgorithmSettings._check_default_settings(settings)
        self.name = self._get_name(settings)
        self.parameters = self._get_parameters(settings)
        self.seed = self._get_seed(settings)
        self.loss = self._get_loss(settings)

    @staticmethod
    def _check_default_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError("The 'name' key is missing in the algorithm settings (JSON file) you are loading")
        if 'seed' not in settings.keys():
            raise ValueError("The 'settings' key is missing in the algorithm settings (JSON file) you are loading")
        if 'parameters' not in settings.keys():
            raise ValueError("The 'parameters' key is missing in the algorithm settings (JSON file) you are loading")
        if 'initialization_method' not in settings.keys():
            raise ValueError(
                "The 'initialization_method' key is missing in the algorithm settings (JSON file) you are loading")
        if 'loss' not in settings.keys():
            warnings.warn("The 'loss' key is missing in the algorithm settings (JSON file) you are loading. \
            Its value will be 'MSE' by default")

    @staticmethod
    def _get_name(settings):
        return settings['name'].lower()

    @staticmethod
    def _get_parameters(settings):
        return settings['parameters']

    @staticmethod
    def _get_seed(settings):
        if settings['seed'] is None:
            return None
        try:
            return int(settings['seed'])
        except ValueError:
            print("The 'seed' parameter you provided cannot be converted to int")

    @staticmethod
    def _get_initialization_method(settings):
        if settings['initialization_method'] is None:
            return None
        # TODO : There should be a list of possible initialization method. It can also be discussed depending on the algorithms name
        return settings['initialization_method']

    @staticmethod
    def _get_loss(settings):
        if 'loss' not in settings.keys():
            # Return default value for the loss (Mean Squared Error)
            return 'MSE'
        if settings['loss'] in ['MSE', 'MSE_diag_noise', 'crossentropy']:
            return settings['loss']
        else:
            raise ValueError("The loss provided is not recognised. Should be one of ['MSE', 'MSE_diag_noise', 'crossentropy']")


