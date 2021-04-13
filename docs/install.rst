.. _install:

Installation & testing
**********************

Dependencies
------------

`leaspy` requires:

- Python (>=3.6)
- numpy (>=1.16.6)
- scipy (>=1.5.4)
- scikit-learn (>=0.21.3, <0.24)
- pandas (>=1.0.5)
- torch (>=1.2.0, <1.7)
- joblib (>=0.13.2)
- matplotlib>=3.0.3
- statsmodels (>=0.12.1)


User installation
-----------------

1. (Optional) Create a dedicated `conda environment`::

    conda create --name leaspy python=3.7
    conda activate leaspy


2. Download `leaspy` with `pip`::

    pip install leaspy


Testing
-------

After installation, you can run the examples in :ref:`nutshell` and in :ref:`the Leaspy API <api>`.
To do so, in your `leaspy environment`, you can download ``ipykernel`` to use `leaspy` with `jupyter`::

    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=leaspy

Now, you can open `jupyter lab` or `jupyter notebook` and select the `leaspy kernel`.


.. Development
.. -----------
