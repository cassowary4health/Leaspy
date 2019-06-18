#%%


import os
from src.main import Leaspy
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings


leaspy = Leaspy("univariate")

#%%

test_data_dir = os.path.join("../tests", "_data")

# Load Data
data_path = os.path.join(test_data_dir, 'univariate_data.csv')
reader = DataReader()
data = reader.read(data_path)

# Path algo
path_to_algorithm_settings = os.path.join(test_data_dir,
                                          '_fit_univariatesigmoid_gradientdescent',
                                          "algorithm_settings.json")
AlgoSettings = AlgoSettings(path_to_algorithm_settings)
AlgoSettings.output_path = "../../output_leaspy/usecase/"

#%%


leaspy.fit(data, AlgoSettings)
leaspy.save("../../output_leaspy/usecase/model_usecase.json")




#%% Relaunch with previous settings

leaspy2 = Leaspy.from_model_settings("../output_leaspy/usecase/model_usecase.json")
leaspy2.fit(data, AlgoSettings)