.. _install:

Installation & testing
**********************

Dependencies
------------

`leaspy` requires:

- Python (>= 3.5)
- numpy (>=1.16.2)
- scipy (>=1.2.1)
- scikit-learn (>=0.21.3)
- pandas (==0.24.2)
- torch (>=1.1.0, <1.5)
- joblib (>=0.10)


User installation
-----------------

1. (Optional) Create a dedicated `conda environment`::

    conda create --name leaspy python=3.7
    conda activate leaspy


2. Download `leaspy` with `pip`::

    pip install leaspy


Testing
-------

After installation, you can run the examples in :ref:`nutshell` and in :ref:`the Leaspi API <leaspy_api>`.
To do so, in your `leaspy environment`, you can download ``ipykernel`` to use `leaspy` with `jupyter`::

    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=leaspy_tutorial

Now, you can open `jupyter lab` or `jupyter notebook` and select the `leaspy kernel`.


.. Development
.. -----------
