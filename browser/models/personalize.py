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


    # Scores
    scores = [_[1:] for _ in data['scores']]
    scores = pd.DataFrame(data=scores, columns=['a', 'b', 'c', 'd'])
    scores['ID'] = 1.
    scores['TIME'] = ages


    return Data.from_dataframe(scores)

def get_individual_parameters(data):
    # Data
    birthday = data['birthday']
    table = data['scores']

    leaspy_data = convert_data(data)

    # Algorithm
    settings = AlgorithmSettings('scipy_minimize')

    # Leaspy
    #leaspy = Leaspy.load(data['model'])
    # TO CORRECT
    if data['model']['name'] == 'logistic_parallel':
        leaspy = Leaspy.load(os.path.join(os.getcwd(), 'data', 'example', 'logistic_parallel_parameters.json'))
    elif data['model']['name'] == 'logistic':
        leaspy = Leaspy.load(os.path.join(os.getcwd(), 'data', 'example', 'logistic_parameters.json'))
    result = leaspy.personalize(leaspy_data, settings=settings)

    individual_parameters = {
        'xi': result.individual_parameters['xi'].numpy().tolist(),
        'tau': result.individual_parameters['tau'].numpy().tolist(),
        'sources': result.individual_parameters['sources'].numpy().tolist()[0]
    }

    output = {
        'individual_parameters' : individual_parameters,
        'scores': leaspy_data.to_dataframe().values.T.tolist()
    }

    return output
