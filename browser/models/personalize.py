import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Add leaspy source to path (overwrite any existing leaspy package by inserting instead of appending)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from leaspy import Leaspy, Data, AlgorithmSettings


def non_float_to_nan(obj):
    try:
        return float(obj)
    except Exception:
        return np.nan

def convert_data(data):

    # Ages
    birthday = datetime.strptime(data['birthday'], '%Y-%m-%d')
    dates = [_[0] for _ in data['scores'] if _[0]]
    dates = [datetime.strptime(_, '%m/%d/%Y') for _ in dates]
    ages = [relativedelta(_, birthday) for _ in dates]
    ages = [_.years + _.months/12 + _.days/365 for _ in ages]
    ages = np.array(ages, dtype=np.float32)

    # Scores
    scores = [list(map(non_float_to_nan, _[1:])) for _ in data['scores'] if _[0]]
    scores = np.array(scores, dtype=np.float32)
    scores = pd.DataFrame(data=scores,
                          columns=data['model']['features'])
    scores['ID'] = "patient"
    scores['TIME'] = ages

    scores = scores.set_index(['ID', 'TIME']).dropna(how='all')
    assert scores.index.is_unique, "Patient's ages are not unique..."

    return Data.from_dataframe(scores)

def get_individual_parameters(data):
    # Data
    leaspy_data = convert_data(data)

    # Algorithm
    settings = AlgorithmSettings('scipy_minimize', seed=0, progress_bar=False, use_jacobian=True)

    # Remove non-supported model hyperparameters
    if data['model']['name'].startswith('univariate_'):
        data['model'].pop('dimension')
        data['model'].pop('source_dimension')

    # Leaspy loading + personalization
    leaspy = Leaspy.load(data['model'])
    individual_parameters = leaspy.personalize(leaspy_data, settings=settings)

    # Replace nans by None to be JSON-compliant
    df = leaspy_data.to_dataframe().set_index('ID').fillna(np.nan)
    df = df.replace([np.nan], [None])

    output = {
        'individual_parameters': individual_parameters["patient"],
        'scores': df.to_dict(orient='list')
    }

    return output
