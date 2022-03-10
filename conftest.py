"""This file is intended to configure Pytest before any test."""

# pylint: disable=import-outside-toplevel, protected-access, unused-argument


def pytest_configure(config):
    """
    Configure stuff to know that we are running a Pytest session.

    Parameters
    ----------
    config : ?
        Pytest config
    """
    import sys

    sys._called_from_test = True


def pytest_unconfigure(config):
    """
    Un-configure stuff after running a Pytest session.

    Parameters
    ----------
    config : ?
        Pytest config
    """
    import sys

    del sys._called_from_test
