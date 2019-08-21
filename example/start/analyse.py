import os
from leaspy.main import Leaspy
from leaspy.inputs.data.data import Data

data = Data.from_csv_file(os.path.join(os.path.dirname(__file__), '_inputs', 'data_tiny.csv'))

path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy = Leaspy.load(path_to_saved_model)

path_to_individual_parameters = os.path.join(os.path.dirname(__file__), '_outputs', 'individual_parameters.json')
individual_parameters = leaspy.load_individual_parameters(path_to_individual_parameters)

print(individual_parameters)
