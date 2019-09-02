import unittest

from tests.inputs.model_settings import ModelSettingsTest
from tests.inputs.algorithm_settings import AlgorithmSettingsTest
from tests.inputs.data.individual_data import IndividualDataTest
from tests.inputs.data.data import DataTest
from tests.inputs.data.data_reader import DataReaderTest
from tests.inputs.data.dataset import DatasetTest

from tests.models.abstract_model import AbstractModelTest
from tests.models.utils.attributes import AttributesTest
#from tests.models.univariate_model import UnivariateModelTest
#from tests.models.multivariate_model import MultivariateModelTest
#from tests.models.model_factory import ModelFactoryTest

#from tests.samplers.sampler import SamplerTest
#from tests.samplers.random_variable.gaussian_random_variable import GaussianRandomVariableTest

#from tests.main.main import LeaspyTest
from tests.main.main_fit import LeaspyFitTest
from tests.main.main_personalize import LeaspyPersonalizeTest

unittest.main()