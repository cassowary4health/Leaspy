import pandas as pd

from leaspy.io.data.dataframe_data_reader import DataframeDataReader


class CSVDataReader(DataframeDataReader):
    """
    Methods to convert `csv files` to `Leaspy`-compliant data containers.

    Only a wrapper for :func:`pandas.read_csv` and :class:`.DataframeDataReader`.

    Parameters
    ----------
    path : str
        Path to the csv file to load (with its extension)
    pd_read_csv_kws : kwargs
        Keyword arguments passed to :func:`pandas.read_csv`
    **df_reader_kws
        Keyword arguments passed to :class:`.DataframeDataReader`

    Raises
    ------
    :exc:`.LeaspyDataInputError`
    """
    def __init__(self, path: str, *, pd_read_csv_kws: dict = {}, **df_reader_kws):

        # enforce ID to be interpreted as string as default (can be overwritten)
        pd_read_csv_kws = {'dtype': {'ID': str}, **pd_read_csv_kws}

        df = pd.read_csv(path, **pd_read_csv_kws)
        super().__init__(df, **df_reader_kws)
