import csv

from src.inputs.data.individual_data import IndividualData


class DataReader:
    def __init__(self, path):
        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.n_individuals = 0
        self.n_visits = 0
        self.n_observations = 0

        self._read(path)

    def _check_headers(self, csv_headers):
        if len(csv_headers) < 3:
            raise ValueError("There must be at least three columns in the input dataset")
        if csv_headers[0].lower() != 'id':
            raise ValueError("The first column of the input csv must be \'ID\'")
        if csv_headers[1].lower() != 'time':
            raise ValueError("The second column of the input csv must be \'Time\'")

        self.headers = csv_headers[2:]

    def _get_timepoint(self, idx, timepoint):
        try:
            return float(timepoint)
        except ValueError:
            print('The timepoint {} of individual {} cannot be converted to float'.format(timepoint, idx))

    def _get_observation(self, idx, timepoint, observation):
        try:
            return [float(_) for _ in observation]
        except ValueError:
            print('The observations of individual {} at time {} cannot be converted to float').format(idx, timepoint)

    def _read(self, path):
        # Read csv
        with open(path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_headers = next(csv_reader)
            self._check_headers(csv_headers)

            # Add new individuals
            for row in csv_reader:
                idx = row[0]
                timepoint = self._get_timepoint(idx, row[1])
                observation = self._get_observation(idx, timepoint, row[2:])

                if idx not in self.individuals:
                    self.individuals[idx] = IndividualData(idx)
                    self.iter_to_idx[self.n_individuals] = idx
                    self.n_individuals += 1

                self.individuals[idx].add_observation(timepoint, observation)
                self.n_visits += 1
                self.n_observations += len(observation)

    '''
    def read(self, path):
        self.check_headers(path)
        df = pd.read_csv(path, index_col=['ID', 'TIME'])

        df, time_mean, time_std, time_min, time_max = self.normalize_time(df)
        dimension = df.shape[1]

        data = Data()
        data.set_time_normalization_info(time_mean, time_std, time_min, time_max)
        data.set_dimension(dimension)

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

        time_min = df['TIME'].min()
        time_max = df['TIME'].max()

        df.set_index(['ID', 'TIME'], inplace=True)

        return df, time_mean, time_std, time_min, time_max
    '''
