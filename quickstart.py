import os

"""
import pandas as pd
data_path = os.path.join("../data_leaspy/", "data_adas",'df.csv')
df = pd.read_csv(data_path).iloc[:,[0,1,6]]
df.columns = ['ID','AGE','Value']
df.set_index(['ID','AGE'], inplace=True)
data_dir = "../data_leaspy/quickstart"


import numpy as np
age_mean = np.mean(df.index.get_level_values(level=1))
age_std = np.std(df.index.get_level_values(level=1))

df.reset_index(inplace=True)
df['AGE']= (df['AGE']-age_mean)/age_std
df.set_index(['ID', 'AGE'])

df.to_csv(os.path.join(data_dir, 'df_univariate.csv'))


import pandas as pd
data_dir = "example/multivariate/"
data_path = os.path.join(data_dir, 'data.csv')

df = pd.read_csv(data_path)
df.set_index(['ID','TIME'], inplace=True)
df = df.loc[df.index.drop_duplicates(keep=False)]
df.to_csv(os.path.join(data_dir, 'data2.csv'))

"""


#%%


#%%

import os
from src.main import Leaspy
from src.inputs.data_reader import DataReader
from src.inputs.algo_settings import AlgoSettings

# Settings
#data_dir = "../data_leaspy/quickstart"
data_dir = "example/multivariate/"
path_to_model_parameters = os.path.join(data_dir, 'model_parameters.json')
path_to_algo_parameters = os.path.join(data_dir, "algorithm_settings.json")
algo_settings = AlgoSettings(path_to_algo_parameters)

# Output folder
path_output = '../output_leaspy/quickstart/'
if not os.path.exists(path_output):
    if not os.path.exists('../output_leaspy'):
        os.mkdir('../output_leaspy')
    os.mkdir(path_output)

algo_settings.output_path = path_output

# Leaspy instanciate
#leaspy = Leaspy.from_model_settings(path_to_model_parameters)
leaspy = Leaspy('multivariate')

# Load the data
data_path = os.path.join(data_dir, 'data2_tiny.csv')
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

