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
        df = pd.read_csv(path, index_col=['ID', 'TIME'])

        df, time_mean, time_std = self.normalize_time(df)

        data = Data()
        data.set_time_normalization_info(time_mean, time_std)

        for idx in df.index.get_level_values(0).unique():
            individual = IndividualData(idx)
            for time, values in df.loc[idx].iterrows():
                individual.add_observation(time, values.values)

            data.add_individual(individual)

        return data


    def check_headers(self, path):
        with open(path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_headings = next(csv_reader)

        if 'ID' not in csv_headings:
            raise ValueError('Your csv file with the individual data has to contain an \'ID\' column')
        if 'TIME' not in csv_headings:
            raise ValueError('Your csv file with the individual data has to contain an \'TIME\' column')

        self.headers = [_ for _ in csv_headings if _ not in ['ID', 'TIME']]

    def normalize_time(self, df):
        df.reset_index(inplace=True)

        time_mean = df['TIME'].mean()
        time_std = df['TIME'].std()

        df['TIME'] = (df['TIME']-time_mean)/time_std

        df.set_index(['ID', 'TIME'], inplace=True)

        return df, time_mean, time_std
