import os

test_data_dir = os.path.join(os.path.dirname(__file__), "_data")
default_models_dir = os.path.join(os.path.dirname(__file__), "../leaspy/models/data")
default_algo_dir = os.path.join(os.path.dirname(__file__), "../leaspy/algo/data")

example_data_path = os.path.join(os.path.dirname(__file__), "_data/io/data/data_tiny.csv")
binary_data_path = os.path.join(os.path.dirname(__file__), "_data/io/data/binary_data.csv")
example_logisticmodel_path = os.path.join(os.path.dirname(__file__), "_data/model_parameters/fitted_multivariate_model.json")
example_logisticmodel_diag_noise_path = os.path.join(os.path.dirname(__file__), "_data/model_parameters/fitted_multivariate_model_diag_noise.json")
