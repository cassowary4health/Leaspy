import csv
import pandas as pd

from src.inputs.individual_data import IndividualData
from src.inputs.data import Data

class DataReader():
    """
    Read a csv file with patients data and convert it to model inputs
    """
    def __init__(self):
        self.headers = None

    def read(self, path):
        self.check_headers(path)
        df = pd.read_csv(path, index_col=['ID', 'AGE'])
        data = Data()

        for idx in df.index.get_level_values(0).unique():
            individual = IndividualData(idx)
            for age, values in df.loc[idx].iterrows():
                individual.add_observation(age, values.values)

            data.add_individual(individual)

        return data


    def check_headers(self, path):
        with open(path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_headings = next(csv_reader)

        if 'ID' not in csv_headings:
            raise ValueError('Your csv file with the individual data has to contain an \'ID\' column')
        if 'AGE' not in csv_headings:
            raise ValueError('Your csv file with the individual data has to contain an \'AGE\' column')

        self.headers = [_ for _ in csv_headings if _ not in ['ID', 'AGE']]