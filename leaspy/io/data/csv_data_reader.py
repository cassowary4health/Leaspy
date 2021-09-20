import csv

from leaspy.io.data.individual_data import IndividualData
from leaspy.exceptions import LeaspyDataInputError


class CSVDataReader:
    """
    Methods to convert `csv files` to data containers `Leaspy` compliants.

    Raises
    ------
    LeaspyDataInputError
    """
    def __init__(self, path):
        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0

        self._read(path)

    def _check_headers(self, csv_headers):
        if len(csv_headers) < 3:
            raise LeaspyDataInputError("There must be at least three columns in the input dataset")
        if csv_headers[0].upper() != 'ID':
            raise LeaspyDataInputError("The first column of the input csv must be 'ID'")
        if csv_headers[1].upper() != 'TIME':
            raise LeaspyDataInputError("The second column of the input csv must be 'TIME'")

        self.headers = csv_headers[2:]

    @staticmethod
    def _get_timepoint(idx, timepoint):
        if timepoint != timepoint:
            raise LeaspyDataInputError(f"One of the time value of individual '{idx}' is NaN")
        try:
            return float(timepoint)
        except Exception:
            raise LeaspyDataInputError(f"The timepoint '{timepoint}' of individual '{idx}' cannot be converted to float")

    @staticmethod
    def _get_observation(idx, timepoint, observation):
        try:
            return [float(_) for _ in observation]
        except Exception:
            raise LeaspyDataInputError(f"The observations of individual '{idx}' at time '{timepoint}' cannot be converted to float")

    def _check_observation(self, observation):
        if self.dimension is None:
            self.dimension = len(observation)
        elif len(observation) != self.dimension:
            raise LeaspyDataInputError(f'Number of features mismatch: {len(observation)} != {self.dimension}')

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
                if observation is not None:
                    self._check_observation(observation)

                    if idx not in self.individuals:
                        self.individuals[idx] = IndividualData(idx)
                        self.iter_to_idx[self.n_individuals] = idx
                        self.n_individuals += 1

                    self.individuals[idx].add_observation(timepoint, observation)
                    self.n_visits += 1

