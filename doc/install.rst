.. _install:

Installation & testing
**********************

Dependencies
------------

Leaspy requires:

- Python (>= 3.5)
- numpy (>=1.16.2)
- scipy (>=1.2.1)
- scikit-learn (>=0.21.3)
- pandas (==0.24.2)
- torch (>=1.1.0, <1.5)
- joblib (>=0.10)


User installation
-----------------

You can get the latest version of leaspy by cloning the repository::

    git clone https://gitlab.com/icm-institute/aramislab/leaspy.git
    cd leaspy
    pip install -r requirements.txt


Testing
-------

After installation, you can launch the test suite from inside the source
directory using ``unittest``::

    python -m unittest discover


.. Development
.. -----------
