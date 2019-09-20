import unittest

### Unit tests ###

# Inputs
from tests.unit_tests.inputs.data.data import DataTest
from tests.unit_tests.inputs.data.dataframe_data_reader import DataframeDataReaderTest
from tests.unit_tests.inputs.data.csv_data_reader import CSVDataReaderTest
from tests.unit_tests.inputs.data.individual_data import IndividualDataTest
from tests.unit_tests.inputs.data.dataset import DatasetTest
from tests.unit_tests.inputs.data.result import ResultTest

from tests.unit_tests.inputs.settings.algorithm_settings import AlgorithmSettingsTest
from tests.unit_tests.inputs.settings.model_settings import ModelSettingsTest
from tests.unit_tests.inputs.settings.outputs_settings import OutputSettingsTest

# Models
from tests.unit_tests.models.abstract_model import AbstractModelTest
from tests.unit_tests.models.model_factory import ModelFactoryTest

# Algorithms

# Utils




### Functional tests ###
from tests.functional_tests.api.api_fit import LeaspyFitTest
from tests.functional_tests.api.api_personalize import LeaspyPersonalizeTest
from tests.functional_tests.api.api import LeaspyTest



### Run ###
unittest.main()