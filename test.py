import unittest

from tests.inputs.model_parameters_reader import ModelParametersReaderTest
from tests.inputs.data_reader import DataReaderTest
from tests.inputs.individual_data import IndividualDataTest
from tests.inputs.data import DataTest

from tests.models.model_factory import ModelFactoryTest
from tests.models.abstract_model import AbstractModelTest
from tests.models.univariate_model import UnivariateModelTest

from tests.utils.sampler import SamplerTest
from tests.utils.random_variable.gaussian_random_variable import GaussianRandomVariableTest

# Main
from tests.main.main import LeaspyTest
from tests.main.main_fit import LeaspyFitTest


unittest.main()