import os
import json
import warnings

from leaspy.inputs.settings.outputs_settings import OutputsSettings
from leaspy.inputs.settings import default_data_dir

class AlgorithmSettings:
    """
    Read a algo_parameters json file and create the corresponding algo
    """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parameters = None
        self.seed = None
        self.logs = None

        if name in ['mcmc_saem', 'scipy_minimize']:
            self.load_json(os.path.join(default_data_dir, 'default_' + name + '.json'))
        else:
            raise ValueError('The algorithm name {} you provided does not exist'.format(name))

        for k, v in kwargs.items():
            if k in self.parameters.keys():

                self.parameters[k] = v
            else:
                warnings.warn("The parameter key you provided is unknown")

        # Todo : Make it better
        if name == 'mcmc_saem':
            if 'n_iter' in kwargs.keys() and 'n_burn_in_iter' not in kwargs.keys():
                self.parameters['n_burn_in_iter'] = int(0.9 * kwargs['n_iter'])


    @classmethod
    def load(cls, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        algorithm_settings = cls(settings['name'])
        if 'name' in settings.keys():
            algorithm_settings.name = cls._get_name(settings)

        if 'parameters' in settings.keys():
            algorithm_settings.parameters = cls._get_name(settings)

        if 'seed' in settings.keys():
            algorithm_settings.seed = cls._get_seed(settings)


    def load_json(self, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        AlgorithmSettings._check_settings(settings)
        self.name = self._get_name(settings)
        self.parameters = self._get_parameters(settings)
        self.seed = self._get_seed(settings)

    def set_logs(self, path, **kwargs):
        settings = {
            'path': path,
            'console_print_periodicity': 50,
            'plot_periodicity': 100,
            'save_periodicity': 50
        }

        for k, v in kwargs.items():
            if k in ['console_print_periodicity', 'plot_periodicity', 'save_periodicity']:
                # Todo : Ca devrait planter si v n'est pas un int!!
                settings[k] = v
            else:
                warnings.warn("The kwargs {} you provided is not correct".format(v))

        self.logs = OutputsSettings(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'name\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the algorithm settings (JSON file) you are loading')

    @staticmethod
    def _get_name(settings):
        return settings['name'].lower()

    @staticmethod
    def _get_parameters(settings):
        return settings['parameters']

    @staticmethod
    def _get_seed(settings):
        if 'seed' in settings.keys():
            try:
                return int(settings['seed'])
            except ValueError:
                print("The \'seed\' parameter you provided cannot be converted to int")
        else:
            return None


