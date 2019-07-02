import os
from src.main import Leaspy
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings

# Settings
project_dir = os.path.join(os.path.dirname(__file__), '..')



#%%
data_dir = os.path.join(project_dir, "example/data/multivariate/")
path_to_model_parameters = os.path.join(data_dir, 'model_parameters.json')
path_to_algo_parameters = os.path.join(data_dir, "algorithm_settings.json")
algo_settings = AlgoSettings(path_to_algo_parameters)

# Output folder
path_output = os.path.join(project_dir, '../output_leaspy/quickstart/')

if not os.path.exists(path_output):
    raise ValueError("output folder {0} does not exists yet".format(path_output))

algo_settings.output_path = path_output


# Leaspy instanciate
#leaspy = Leaspy.from_model_settings(path_to_model_parameters)
leaspy = Leaspy('multivariate')

# Load the data
data_path = os.path.join(data_dir, 'data2_tiny.csv')
#data_path = os.path.join(data_dir, 'data2.csv')
reader = DataReader()
data = reader.read(data_path)

#%%
#import pandas as pd
#df = pd.read_csv(os.path.join(data_dir, 'data2.csv'))
#df = df.iloc[:200,:]
#df.set_index(['ID','TIME']).to_csv(os.path.join(data_dir, 'data2_tiny.csv'))

#%%

# Fit
leaspy.fit(data, algo_settings, seed=0)

