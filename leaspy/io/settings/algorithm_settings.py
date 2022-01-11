import json
import os
import warnings

from leaspy.io.settings import algo_default_data_dir
from leaspy.io.settings.outputs_settings import OutputsSettings
from leaspy.algo.algo_factory import AlgoFactory

from leaspy.exceptions import LeaspyAlgoInputError
from leaspy.utils.typing import KwargsType, Optional


class AlgorithmSettings:
    """
    Used to set the algorithms' settings.
    All parameters, except the choice of the algorithm, is set by default.
    The user can overwrite all default settings.

    Parameters
    ----------
    name : str
        The algorithm's name. Must be in:
            * For `fit` algorithms:
                * ``'mcmc_saem'``
                * ``'lme_fit'`` (for LME model only)
            * For `personalize` algorithms:
                * ``'scipy_minimize'``
                * ``'mean_real'``
                * ``'mode_real'``
                * ``'constant_prediction'`` (for constant model only)
                * ``'lme_personalize'`` (for LME model only)
            * For `simulate` algorithms:
                * ``'simulation'``

    **kwargs : any
        Depending on algorithm you want to personalize, various parameters are possible (not exhaustive):
            * model_initialization_method : str, optional
                For **fit** algorithms only, give a model initialization method,
                according to those possible in :func:`~.models.utils.initialization.model_initialization.initialize_parameters`.
            * algo_initialization_method : str, optional
                Personalize the algorithm initialization method,
                according to those possible for the given algorithm (refer to its documentation in :mod:`leaspy.algo`).
            * n_iter : int, optional
                Number of iteration. There is no stopping criteria for the all the MCMC SAEM algorithms.
            * n_burn_in_iter : int, optional
                Number of iteration during burning phase, used for the MCMC SAEM algorithms.
            * seed : int, optional, default None
                Used for stochastic algorithms.
            * use_jacobian : bool, optional, default True
                Used in ``scipy_minimize`` algorithm to perform a `L-BFGS` instead of a `Powell` algorithm.
            * n_jobs : int, optional, default 1
                Used in ``scipy_minimize`` algorithm to accelerate calculation with parallel derivation using joblib.
            * progress_bar : bool, optional, default True
                Used to display a progress bar during computation.

        For the complete list of the available parameters for a given algorithm, please directly refer to its documentation.

    Attributes
    ----------
    name : str
        The algorithm's name.
    model_initialization_method : str, optional
      For fit algorithms, give a model initialization method,
      according to those possible in :func:`~.models.utils.initialization.model_initialization.initialize_parameters`.
    algo_initialization_method : str, optional
      Personalize the algorithm initialization method,
      according to those possible for the given algorithm (refer to its documentation in :mod:`leaspy.algo`).
    seed : int, optional, default None
      Used for stochastic algorithms.
    parameters : dict
      Contains the other parameters: `n_iter`, `n_burn_in_iter`, `use_jacobian`, `n_jobs` & `progress_bar`.
    logs : :class:`.OutputsSettings`, optional
      Used to create a ``logs`` file during a model calibration containing convergence information.

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`

    See Also
    --------
    :mod:`leaspy.algo`

    Notes
    -----
    For developers: use ``_dynamic_default_parameters`` to dynamically set some default parameters,
    depending on other parameters that were set, while these `dynamic` parameters were not set.

    Example:
        you could want to set burn in iterations or annealing iterations
        as fractions of non-default number of iterations given.

    Format:

    ::

        {algo_name: [
            (functional_condition_to_trigger_dynamic_setting(kwargs),
            {
                nested_keys_of_dynamic_setting: dynamic_value(kwargs)
            })
        ]}
    """

    # TODO should be in the each algo class directly?
    _dynamic_default_parameters = {
        'mcmc_saem': [
            (
                # a number of iteration is given
                lambda kw: 'n_iter' in kw,
                {
                    # burn-in: 90% of iterations given
                    ('n_burn_in_iter',): lambda kw: int(0.9 * kw['n_iter']),
                    # annealing: 50% of iterations given
                    ('annealing', 'n_iter'): lambda kw: int(0.5 * kw['n_iter'])
                }
            )
        ],

        'lme_fit': [
            (
                lambda kw: 'force_independent_random_effects' in kw and kw['force_independent_random_effects'],
                {
                    ('method',): lambda kw: ['lbfgs','bfgs'] # powell & nm methods cannot ensure respect of "free"
                }
            )
        ]
    }

    # known keys for all algorithms (<!> not all of them are mandatory!)
    _known_keys = ['name', 'seed', 'algorithm_initialization_method', 'model_initialization_method', 'parameters'] # 'logs' are not handled in exported files

    def __init__(self, name: str, **kwargs):

        self.name = name
        self.parameters: KwargsType = None # {}
        self.seed: Optional[int] = None
        self.algorithm_initialization_method: str = None # Initialization of the algorithm itself
        self.model_initialization_method: str = None # Initialization of the model parameters (independently of the algorithm, for fit family only)
        self.logs = None

        default_algo_settings_path = os.path.join(algo_default_data_dir, 'default_' + name + '.json')

        if os.path.isfile(default_algo_settings_path):
            self._load_default_values(default_algo_settings_path)
        else:
            raise LeaspyAlgoInputError(f"The algorithm name '{name}' you provided does not exist")

        self._manage_kwargs(kwargs)
        self.check_consistency()

    @property
    def algo_class(self):
        """Class of the algorithm derived from its name (shorthand)."""
        return AlgoFactory.get_class(self.name)

    def check_consistency(self) -> None:
        """
        Check internal consistency of algorithm settings and warn or raise a `LeaspyAlgoInputError` if not.
        """
        algo_family = self.algo_class.family
        if self.model_initialization_method is not None and algo_family != 'fit':
            warnings.warn('You should not define `model_initialization_method` in your algorithm: '
                          f'"{self.name}" is not a `fit` algorithm so it has no effect.')

        if self.seed is not None and self.algo_class.deterministic:
            warnings.warn(f'You can skip defining `seed` since the algorithm {self.name} is deterministic.')

    @classmethod
    def load(cls, path_to_algorithm_settings: str):
        """
        Instantiate a AlgorithmSettings object a from json file.

        Parameters
        ----------
        path_to_algorithm_settings : str
            Path of the json file.

        Returns
        -------
        :class:`.AlgorithmSettings`
            An instanced of AlgorithmSettings with specified parameters.

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            if anything is invalid in algo settings

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
        >>> leaspy_univariate = AlgorithmSettings.load('outputs/leaspy-univariate_model-settings.json')
        """
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        if 'name' not in settings.keys():
            raise LeaspyAlgoInputError("Your json file must contain a 'name' attribute!")

        algorithm_settings = cls(settings['name'])

        if 'parameters' in settings.keys():
            print("You overwrote the algorithm default parameters")
            algorithm_settings.parameters = cls._get_parameters(settings)

        if 'seed' in settings.keys():
            print("You overwrote the algorithm default seed")
            algorithm_settings.seed = cls._get_seed(settings)

        if 'algorithm_initialization_method' in settings.keys():
            print("You overwrote the algorithm default initialization method")
            algorithm_settings.algorithm_initialization_method = cls._get_algorithm_initialization_method(settings)

        if 'model_initialization_method' in settings.keys():
            print("You overwrote the model default initialization method")
            algorithm_settings.model_initialization_method = cls._get_model_initialization_method(settings)

        if 'loss' in settings.keys():
            raise LeaspyAlgoInputError('`loss` keyword for AlgorithmSettings is not supported any more. '
                                       'Please define `noise_model` directly in your Leaspy model.')

        # Unknown keys
        # TODO: this class should really be refactored so not to copy in 3 methods same stuff (manage_kwargs, load & _check_default_settings)
        unknown_keys = set(settings.keys()).difference(cls._known_keys)
        if unknown_keys:
            raise LeaspyAlgoInputError(f'Unexpected keys {unknown_keys} in algorithm settings.')

        algorithm_settings.check_consistency()

        return algorithm_settings

    def save(self, path: str, **kwargs):
        """
        Save an AlgorithmSettings object in a json file.

        TODO? save leaspy version as well for retro/future-compatibility issues?

        Parameters
        ----------
        path : str
            Path to store the AlgorithmSettings.
        **kwargs
            Keyword arguments for json.dump method.
            Default: dict(indent=2)

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
        >>> settings = AlgorithmSettings('scipy_minimize', seed=42, n_jobs=-1, use_jacobian=True, progress_bar=True)
        >>> settings.save('outputs/scipy_minimize-settings.json')
        """
        json_settings = {
            "name": self.name,
            "seed": self.seed,
            "algorithm_initialization_method": self.algorithm_initialization_method,
        }

        algo_family = self.algo_class.family
        if algo_family == 'fit':
            json_settings['model_initialization_method'] = self.model_initialization_method

        """
        TODO: save config of logging as well (OutputSettings needs to be JSON serializable...)
        if self.logs is not None:
            json_settings['logs'] = self.logs
        """

        # append parameters key after "hyperparameters"
        json_settings['parameters'] = self.parameters

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(os.path.join(path), "w") as json_file:
            json.dump(json_settings, json_file, **kwargs)

    def set_logs(self, path, **kwargs):
        """
        Use this method to monitor the convergence of a model callibration.

        It create graphs and csv files of the values of the population parameters (fixed effects) during the callibration

        Parameters
        ----------
        path : str
            The path of the folder to store the graphs and csv files.
        **kwargs
            * console_print_periodicity: int, optional, default 50
                Display logs in the console/terminal every N iterations.
            * plot_periodicity: int, optional, default 100
                Saves the values to display in pdf every N iterations.
            * save_periodicity: int, optional, default 50
                Saves the values in csv files every N iterations.
            * overwrite_logs_folder: bool, optional, default False
                Set it to ``True`` to overwrite the content of the folder in ``path``.

        Raises
        ------
        :exc:`.LeaspyAlgoInputError`
            If the folder given in ``path`` already exists and if ``overwrite_logs_folder`` is set to ``False``.

        Notes
        -----
        By default, if the folder given in ``path`` already exists, the method will raise an error.
        To overwrite the content of the folder, set ``overwrite_logs_folder`` it to ``True``.
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
                if v is not None and not isinstance(v, int):
                    raise LeaspyAlgoInputError(f'You must provide a integer to the input <{k}>! '
                                    f'You provide {v} of type {type(v)}.')
                settings[k] = v
            elif k in ['overwrite_logs_folder']:
                if not isinstance(v, bool):
                    raise LeaspyAlgoInputError(f'You must provide a boolean to the input <{k}>! '
                                    f'You provide {v} of type {type(v)}.')
                settings[k] = v
            else:
                warnings.warn(f"The kwarg '{k}' you provided is not valid and was skipped.")

        self.logs = OutputsSettings(settings)

    def _manage_kwargs(self, kwargs):

        _special_kwargs = {
            'seed': self._get_seed,
            'algorithm_initialization_method': self._get_algorithm_initialization_method,
            'model_initialization_method': self._get_model_initialization_method,
        }

        for k, v in kwargs.items():

            if k in _special_kwargs:
                k_getter = _special_kwargs[k]
                setattr(self, k, k_getter(kwargs))

            elif k in self.parameters:
                self.parameters[k] = v

            else:
                # TODO: warn with all unknown parameters? (and use `self._known_keys`)
                warnings.warn(f"The parameter '{k}' you provided is unknown and thus was skipped.")

        # dynamic default parameters
        if self.name in self._dynamic_default_parameters:

            for func_condition, associated_defaults in self._dynamic_default_parameters[self.name]:

                if not func_condition(kwargs):
                    continue

                # loop on dynamic defaults
                for nested_levels, val_getter in associated_defaults.items():
                    # check that the dynamic default that we want to set is not already overwritten
                    if self._get_nested_dict(kwargs, nested_levels) is None:
                        self._set_nested_dict(self.parameters, nested_levels, val_getter(kwargs))

    @staticmethod
    def _get_nested_dict(nested_dict: dict, nested_levels, default = None):
        """
        Get a nested key of a dict or default if any previous level is missing.

        Examples
        --------
        >>> _get_nested_dict(d, ('a','b'), -1) == ...
            * -1 if 'a' not in d
            * -1 if 'b' not in d['a']
            * d['a']['b'] else

        >>> _get_nested_dict(d, (), ...) == d
        """
        it_levels = iter(nested_levels)

        while isinstance(nested_dict, dict):
            try:
                next_lvl = next(it_levels)
            except StopIteration:
                break

            # get next level dict
            nested_dict = nested_dict.get(next_lvl, default)

        return nested_dict

    @classmethod
    def _set_nested_dict(cls, nested_dict: dict, nested_levels, val):
        """
        Set a nested key of a dict.
        Precondition: all intermediate levels must exist.
        """
        *nested_top_levels, last_level = nested_levels
        dict_to_set = cls._get_nested_dict(nested_dict, nested_top_levels, default=None)
        assert isinstance(dict_to_set, dict)
        dict_to_set[last_level] = val # inplace

    def _load_default_values(self, path_to_algorithm_settings):

        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        self._check_default_settings(settings)
        # TODO: Urgent => The following function should in fact be algorithm-name specific!! As for the constant prediction
        # Etienne: I'd advocate for putting all non-generic / parametric stuff in special methods / attributes of corresponding algos... so that everything is generic here
        # Igor : Agreed. This class became a real mess.

        self.name = self._get_name(settings)
        self.parameters = self._get_parameters(settings)

        self.algorithm_initialization_method = self._get_algorithm_initialization_method(settings)

        # optional hyperparameters depending on type of algorithm
        algo_class = self.algo_class

        if not algo_class.deterministic:
            self.seed = self._get_seed(settings)

        if algo_class.family == 'fit':
            self.model_initialization_method = self._get_model_initialization_method(settings)

    @classmethod
    def _check_default_settings(cls, settings):

        # Unknown keys
        unknown_keys = set(settings.keys()).difference(cls._known_keys)
        if unknown_keys:
            raise LeaspyAlgoInputError(f'Unexpected keys {unknown_keys} in algorithm settings.')

        # Missing keys
        error_tpl = "The '{}' key is missing in the algorithm settings (JSON file) you are loading."

        for mandatory_key in ['name', 'parameters']:
            if mandatory_key not in settings.keys():
                raise LeaspyAlgoInputError(error_tpl.format(mandatory_key))

        algo_class = AlgoFactory.get_class(settings['name'])

        if not algo_class.deterministic and 'seed' not in settings.keys():
            raise LeaspyAlgoInputError(error_tpl.format('seed'))

        if 'algorithm_initialization_method' not in settings.keys():
            raise LeaspyAlgoInputError(error_tpl.format('algorithm_initialization_method'))

        if algo_class.family == 'fit' and 'model_initialization_method' not in settings.keys():
            raise LeaspyAlgoInputError(error_tpl.format('model_initialization_method'))

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
        except Exception:
            warnings.warn(f"The 'seed' parameter you provided ({settings['seed']}) cannot be converted to int, using None instead.")
            return None

    @staticmethod
    def _get_algorithm_initialization_method(settings):
        if settings['algorithm_initialization_method'] is None:
            return None
        # TODO : There should be a list of possible initialization method. It can also be discussed depending on the algorithms name
        return settings['algorithm_initialization_method']

    @staticmethod
    def _get_model_initialization_method(settings):
        if settings['model_initialization_method'] is None:
            return None
        # TODO : There should be a list of possible initialization method. It can also be discussed depending on the algorithms name
        return settings['model_initialization_method']

