import os
import shutil
import warnings

from leaspy.exceptions import LeaspyAlgoInputError


class OutputsSettings:
    """
    Used to create the `logs` folder to monitor the convergence of the calibration algorithm.

    Raises
    ------
    :class:`.LeaspyAlgoInputError`
    """
    # TODO mettre les variables par défaut à None
    # TODO: Réfléchir aux cas d'usages : est-ce qu'on veut tout ou rien,
    # TODO: ou bien la possibilité d'avoir l'affichage console et/ou logs dans un fold
    # TODO: Aussi, bien définir la création du path

    DEFAULT_LOGS_DIR = '_outputs'  # logs

    def __init__(self, settings):
        self.console_print_periodicity = None
        self.plot_periodicity = None
        self.save_periodicity = None

        self.root_path = None
        self.parameter_convergence_path = None
        self.plot_path = None
        self.patients_plot_path = None

        self._set_console_print_periodicity(settings)
        self._set_plot_periodicity(settings)
        self._set_save_periodicity(settings)
        self._create_root_folder(settings)

    def _set_param_as_int_or_ignore(self, settings, param: str):
        """Inplace set of parameter (as int) from settings."""
        if param not in settings:
            return

        val = settings[param]
        if val is not None:
            # try to cast as an integer.
            try:
                val = int(val)
            except Exception:
                warnings.warn(f"The '{param}' parameters you provided is not castable to an int. "
                              "Ignoring its value.", RuntimeWarning)
                return

        # Update the attribute of self in-place
        setattr(self, param, val)

    def _set_console_print_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'console_print_periodicity')

    def _set_plot_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'plot_periodicity')

    def _set_save_periodicity(self, settings):
        self._set_param_as_int_or_ignore(settings, 'save_periodicity')

    def _create_root_folder(self, settings):
        # Get a path to put the outputs
        if 'path' not in settings.keys():
            warnings.warn("You did not provide a path for your logs outputs. "
                          f"They have been initialized to '{self.DEFAULT_LOGS_DIR}', relatively to the current working directory.")

        rel_or_abs_path = settings.get('path', self.DEFAULT_LOGS_DIR)
        abs_path = os.path.abspath(rel_or_abs_path)

        # store the absolute path in settings
        settings['path'] = abs_path

        # Check if the folder does not exist: if not, create (and its parent)
        if not os.path.exists(abs_path):
            warnings.warn(f"The logs path you provided ({settings['path']}) does not exist. "
                          "Needed paths will be created (and their parents if needed).")
        elif settings['overwrite_logs_folder']:
            warnings.warn(f'Overwrite logs folder...')
            self._clean_folder(abs_path)

        all_ok = self._check_needed_folders_are_empty_or_create_them(abs_path)

        if not all_ok:
            raise LeaspyAlgoInputError("The logs folder already exists and are not empty! "
                    "Give another path or use keyword argument `overwrite_logs_folder=True`.")

    @staticmethod
    def _check_folder_is_empty_or_create_it(path_folder) -> bool:
        if os.path.exists(path_folder):
            if os.path.islink(path_folder) or not os.path.isdir(path_folder) or len(os.listdir(path_folder)) > 0:
                # path is a link, or not a directory, or a directory containing something
                return False
        else:
            os.makedirs(path_folder)

        return True

    @staticmethod
    def _clean_folder(path):
        shutil.rmtree(path)
        os.makedirs(path)

    def _check_needed_folders_are_empty_or_create_them(self, path) -> bool:
        self.root_path = path

        self.parameter_convergence_path = os.path.join(path, 'parameter_convergence')
        self.plot_path = os.path.join(path, 'plots')
        self.patients_plot_path = os.path.join(self.plot_path, 'patients')

        all_ok = self._check_folder_is_empty_or_create_it(self.parameter_convergence_path)
        all_ok &= self._check_folder_is_empty_or_create_it(self.plot_path)
        all_ok &= self._check_folder_is_empty_or_create_it(self.patients_plot_path)

        return all_ok

