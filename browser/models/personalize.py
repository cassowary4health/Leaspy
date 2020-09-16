import os, sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

sys.path.append(os.path.join(os.getcwd(), '..'))


from leaspy import Leaspy, Data, AlgorithmSettings

def convert_data(data):

    # Ages
    birthday = datetime.strptime(data['birthday'], '%Y-%m-%d')
    dates = [_[0] for _ in data['scores']]
    dates = [datetime.strptime(_, '%m/%d/%Y') for _ in dates]
    ages = [relativedelta(_, birthday) for _ in dates]
    ages = [_.years + _.months/12 + _.days/365 for _ in ages]
    ages = np.array(ages, dtype=np.float32)


    # Scores
    scores = [_[1:] for _ in data['scores']]
    scores = np.array(scores, dtype=np.float32)
    scores = pd.DataFrame(data=scores, columns=[str(_) for _ in range(len(scores[0]))])
    scores['ID'] = "patient"
    scores['TIME'] = ages

    return Data.from_dataframe(scores)

def get_individual_parameters(data):
    # Data
    leaspy_data = convert_data(data)

    # Algorithm
    settings = AlgorithmSettings('scipy_minimize')

    # Leaspy

    #leaspy = Leaspy.load(data['model'])
    # TO CORRECT
    #if data['model']['name'] == 'logistic_parallel':
    leaspy = Leaspy.load(data['model'])
    #elif data['model']['name'] == 'logistic':
    #    leaspy = Leaspy.load(os.path.join(os.getcwd(), 'data', 'example', 'parkinson_model.json'))
    individual_parameters = leaspy.personalize(leaspy_data, settings=settings)

    output = {
        'individual_parameters' : individual_parameters["patient"],
        'scores': leaspy_data.to_dataframe().values.T.tolist()
    }

    return output
