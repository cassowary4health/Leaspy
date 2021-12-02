from unittest.mock import patch
from typing import Any, Dict, List
from unittest import TestCase

import numpy as np
import pandas as pd

from leaspy import Data, AlgorithmSettings #, Leaspy,
from tests import example_data_path, binary_data_path


class TestHelpers:

    @staticmethod
    def allow_abstract_class_init(abc_klass):
        """
        Decorator to allow to instantiate an abstract class (for testing only)
        """
        return patch.multiple(abc_klass, __abstractmethods__=set())

    @staticmethod
    def get_data_for_model(model_name: str) -> Data:
        """Helper to load the right data for functional tests, depending on model name."""

        if 'binary' in model_name:
            df = pd.read_csv(binary_data_path)
        else:
            # continuous
            df = pd.read_csv(example_data_path)

        if 'univariate' in model_name:
            df = df.iloc[:, :3] # one feature column (first)

        return Data.from_dataframe(df)

    @staticmethod
    def get_algo_settings(*, path: str = None, name: str = None, **params):
        """Helper to create the AlgorithmSettings object (either from path to saved settings or from name and kwargs)."""
        assert (path is None) ^ (name is None), "Either `path` or `name` should be not None"

        if path is not None:
            return AlgorithmSettings.load(path)
        else:
            return AlgorithmSettings(name, **params)

    @classmethod
    def ordered_containers(cls, obj, *, sort_seqs=(list, set, frozenset)): # tuple
        """Utils function to sort `obj` recursively at all levels (in-depth)."""
        if isinstance(obj, dict):
            return sorted( ((k, cls.ordered_containers(v, sort_seqs=sort_seqs)) for k, v in obj.items()),
                          key=lambda tup: tup[0])
        elif isinstance(obj, sort_seqs):
            return sorted(cls.ordered_containers(x, sort_seqs=sort_seqs) for x in obj)
        else:
            return obj

    @classmethod
    def check_nested_dict_almost_equal(cls, left: dict, right: dict, *,
                                       left_desc: str = 'left', right_desc: str = 'right',
                                       allclose_custom: Dict[str, Dict[str, Any]] = {}, **allclose_defaults) -> List[str]:
        """
        Compare two dictionaries up to customizable tolerances and display their differences if any.

        Parameters
        ----------
        left : dict
        right : dict
            The dictionary to recursively compare

        **allclose_defaults
            Default keyword arguments for `numpy.allclose`:
            * rtol: float = 1e-05
            * atol: float = 1e-08
            * equal_nan: bool = False
        allclose_custom : dict[str, dict[str, Any]] (optional, default None)
            Custom keywords arguments to overwrite default ones, for a particular key (last-level key only)
            e.g. {'noise_std': dict(atol=1e-3), ...}

        left_desc : str
        right_desc : str
            Labels to describe `left` (resp. `right`) in error messages

        Returns
        -------
        list[str]
            Description of ALL reasons why dictionary are NOT equal.
            Empty if and only if ``left`` ~= ``right`` (up to customized tolerances)
        """
        try:
            if left == right:
                return []
        except Exception:
            # in case comparison is not possible
            pass

        if not isinstance(left, dict):
            return [f"`{left_desc}` should be a dictionary"]
        if not isinstance(right, dict):
            return [f"`{right_desc}` should be a dictionary"]

        if left.keys() != right.keys():
            extra_left = [k for k in left.keys() if k not in right.keys()]
            extra_right = [k for k in right.keys() if k not in left.keys()]
            extras = []
            if extra_left:
                extras.append(f'`{left_desc}`: +{extra_left}')
            if extra_right:
                extras.append(f'`{right_desc}`: +{extra_right}')
            nl = '\n'
            return [f"Keys are different:\n{nl.join(extras)}"]

        # loop on keys
        errs = []
        for k, left_v in left.items():
            right_v = right[k]
            # nest key in error messages
            left_k_desc = f'{left_desc}.{k}'
            right_k_desc = f'{right_desc}.{k}'

            if isinstance(left_v, dict) or isinstance(right_v, dict):
                # do not fail early as before
                errs += cls.check_nested_dict_almost_equal(left_v, right_v,
                                                           left_desc=left_k_desc, right_desc=right_k_desc,
                                                           allclose_custom=allclose_custom, **allclose_defaults)
            else:
                # TODO? also nest keys in `allclose_custom`
                # merge keyword arguments for the particular key if any customisation
                allclose_kwds_for_key = {**allclose_defaults, **allclose_custom.get(k, {})}
                cmp_details = ["numpy.allclose"]
                if allclose_kwds_for_key:
                    cmp_details.append(str(allclose_kwds_for_key))
                cmp_suffix = f' ({", ".join(cmp_details)})'

                try:
                    # almost equal?
                    almost_eq = np.allclose(left_v, right_v, **allclose_kwds_for_key)
                except Exception:
                    # truly equal?
                    cmp_suffix = ''
                    almost_eq = bool(left_v == right_v)
                except Exception:
                    # example in case we are trying to compare arrays without the same shape
                    almost_eq = False

                if not almost_eq:
                    # do not fail early as before
                    errs.append(f"Values are different{cmp_suffix}:\n`{left_k_desc}` -> {left_v} != {right_v} <- `{right_k_desc}`")

        # return all error messages if any!
        return errs

    @classmethod
    def assert_nested_dict_almost_equal(cls, t: TestCase, left: dict, right: dict, *,
                                       left_desc: str = 'new', right_desc: str = 'expected',
                                       allclose_custom: Dict[str, Dict[str, Any]] = {}, **allclose_defaults) -> None:

        pbs = cls.check_nested_dict_almost_equal(left, right, left_desc=left_desc, right_desc=right_desc,
                                                 allclose_custom=allclose_custom, **allclose_defaults)

        if pbs:
            t.fail("\n".join(pbs))
