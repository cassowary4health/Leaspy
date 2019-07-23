import os
from src.main import Leaspy
from src.inputs.data.data import Data
from src.inputs.algorithm_settings import AlgorithmSettings
from src.algo.algo_factory import AlgoFactory
from src.inputs.data.dataset import Dataset

# Inputs
data = Data(os.path.join(os.path.dirname(__file__), '_inputs', 'data_tiny.csv'))
algo_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'algorithm_settings.json'))

# Initialize
leaspy = Leaspy("multivariate_parallel")
#leaspy = Leaspy("multivariate")

# Fit the model on the data
leaspy.fit(data, algorithm_settings=algo_settings)

# Save the model
path_to_saved_model = os.path.join(os.path.dirname(__file__), '_outputs', 'fitted_multivariate_model.json')
leaspy.save(path_to_saved_model)

# Personalize
prediction_settings = AlgorithmSettings(os.path.join(os.path.dirname(__file__), '_inputs', 'prediction_settings.json'))
res = leaspy.personalize(data, prediction_settings=prediction_settings)

from src.utils.plotter import Plotter
plotter = Plotter()
algorithm = AlgoFactory.algo(prediction_settings)
dataset = Dataset(data, algo=algorithm, model=leaspy.model)
plotter.plot_patient_reconstructions("example/start/_outputs/predict.pdf", 5 ,dataset, leaspy.model, data.realizations)


# Load the model as if it is another day of your life
#leaspy2 = Leaspy.load(path_to_saved_model)

# Fit a second time
#leaspy2.fit(data, algorithm_settings=algo_settings)
