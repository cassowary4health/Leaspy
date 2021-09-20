import pandas as pd

from leaspy.io.data.individual_data import IndividualData
from leaspy.exceptions import LeaspyDataInputError


class DataframeDataReader:
    """
    Methods to convert :class:`pandas.DataFrame` to data containers `Leaspy` compliants.

    Raises
    ------
    LeaspyDataInputError
    """
    def __init__(self, df: pd.DataFrame):
        self.individuals = {}
        self.iter_to_idx = {}
        self.headers = None
        self.dimension = None
        self.n_individuals = 0
        self.n_visits = 0

        self._read(df)

    @staticmethod
    def _check_headers(columns):
        # cols_upper = list(map(str.upper, columns))
        missing_mandatory_columns = [_ for _ in ['ID', 'TIME'] if _ not in columns]
        if len(missing_mandatory_columns) > 0:
            raise LeaspyDataInputError(f"Your dataframe must have {missing_mandatory_columns} columns")

    def _check_observation(self, observation):
        if self.dimension is None:
            self.dimension = len(observation)
        elif len(observation) != self.dimension:
            raise LeaspyDataInputError(f'Number of features mismatch: {len(observation)} != {self.dimension}')

    def _read(self, df: pd.DataFrame):
        df = df.copy(deep=True)  # No modification on the input dataframe !
        columns = df.columns.values
        # Try to read the raw dataframe
        try:
            self._check_headers(columns)

        # If we do not find 'ID' and 'TIME' columns, check the Index
        except LeaspyDataInputError:
            df.reset_index(inplace=True)
            columns = df.columns.values
            self._check_headers(columns)

        df.set_index(['ID', 'TIME'], inplace=True)
        self.headers = df.columns.values.tolist()

        for (idx, timepoint), v in df.iterrows():
            if timepoint != timepoint:
                raise LeaspyDataInputError(f'One of the time value of individual {idx} is NaN')

            observation = v.values
            self._check_observation(observation)

            if idx not in self.individuals:
                self.individuals[idx] = IndividualData(idx)
                self.iter_to_idx[self.n_individuals] = idx
                self.n_individuals += 1

            self.individuals[idx].add_observation(timepoint, observation)
            self.n_visits += 1

