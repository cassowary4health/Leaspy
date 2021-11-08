import timeit


def setup_performance_fit_logistic():
    return '''
import pandas as pd
import numpy as np
from typing import Optional
def generate_fake_data(
    n_patients:Optional[int]=1000,
    n_visits_per_patient:Optional[int]=5,
    n_features:Optional[int]=3
) -> pd.DataFrame:

    df = pd.DataFrame()
    for i in range(n_patients):
        patient = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, n_features))
        times = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, 1))
        patient_df = pd.DataFrame(patient)
        patient_df.columns = [str(col) for col in patient_df.columns]
        patient_df['ID'] = i
        patient_df['TIME'] = times
        df = pd.concat([df, patient_df])
    df = df[["ID", "TIME"] + [str(i) for i in range(n_features)]]
    return df
    '''


def test_performance_fit_logistic():
    test_performance = '''     
from leaspy import AlgorithmSettings, Data
from tests.functional_tests.api.test_api_fit import LeaspyTestBase
import torch

def leaspy_fit_logistic(): 
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = generate_fake_data()
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_tensor_type('torch.FloatTensor')
    
    algo_settings = AlgorithmSettings(name="mcmc_saem", n_iter=1000, seed=0)
    LeaspyTestBase(model_name="logistic").fit(algo_settings=algo_settings, data=Data.from_dataframe(generate_fake_data()))
    '''
    number_statement = 3
    t = timeit.repeat(stmt=test_performance, setup=setup_performance_fit_logistic(), number=number_statement)

    # For a single iteration exec. time, we need to divide the output time by number.
    result = [time / number_statement for time in t]
    print(result)
