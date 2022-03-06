import os

test_data_dir = os.path.join(os.path.dirname(__file__), "_data")
default_models_dir = os.path.join(os.path.dirname(__file__), "../leaspy/models/data")
default_algo_dir = os.path.join(os.path.dirname(__file__), "../leaspy/algo/data")

example_data_path = os.path.join(os.path.dirname(__file__), "_data/leaspy_io/data/data_tiny.csv")
binary_data_path = os.path.join(os.path.dirname(__file__), "_data/leaspy_io/data/binary_data.csv")

# hardcoded models: good for unit tests & functional tests independent from fit behavior
hardcoded_models_folder = os.path.join(os.path.dirname(__file__), "_data", "model_parameters", "hardcoded")
hardcoded_model_path = lambda model_name: os.path.join(hardcoded_models_folder, model_name + '.json')

# models generated from fit functional tests, bad for most tests as it may change due to slights changes in fit
from_fit_models_folder = os.path.join(os.path.dirname(__file__), "_data", "model_parameters", "from_fit")
from_fit_model_path = lambda model_name: os.path.join(from_fit_models_folder, model_name + '.json')


from unittest.mock import patch

def allow_abstract_class_init(abc_klass):
    """
    Decorator to allow to instantiate an abstract class (for testing only)
    """
    return patch.multiple(abc_klass, __abstractmethods__=set())
