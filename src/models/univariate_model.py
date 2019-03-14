import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_parameters_reader import ModelParametersReader


class UnivariateModel(AbstractModel):
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_univariate_parameters.json")
        reader = ModelParametersReader(data_dir)
        self.model_parameters = reader.parameters
