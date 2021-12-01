import os

# Main directories
test_data_dir = os.path.join(os.path.dirname(__file__), "_data")
default_models_dir = os.path.join(os.path.dirname(__file__), "..", "leaspy", "models", "data")
default_algo_dir = os.path.join(os.path.dirname(__file__), "..", "leaspy", "algo", "data")

# Main mock of data for tests
example_data_path = os.path.join(test_data_dir, "data_mock", "data_tiny.csv")
example_data_covars_path = os.path.join(test_data_dir, "data_mock", "data_tiny_covariate.csv")
binary_data_path = os.path.join(test_data_dir, "data_mock", "binary_data.csv")

# hardcoded models: good for unit tests & functional tests independent from fit behavior
hardcoded_models_folder = os.path.join(test_data_dir, "model_parameters", "hardcoded")
hardcoded_model_path = lambda model_name: os.path.join(hardcoded_models_folder, model_name + '.json')

# models generated from fit functional tests, bad for most tests as it may change due to slights changes in fit
from_fit_models_folder = os.path.join(test_data_dir, "model_parameters", "from_fit")
from_fit_model_path = lambda model_name: os.path.join(from_fit_models_folder, model_name + '.json')

# hardcoded individual parameters: good for unit tests & functional tests independent from personalize behavior
hardcoded_ips_folder = os.path.join(test_data_dir, "individual_parameters", "hardcoded")
hardcoded_ip_path = lambda ip_file: os.path.join(hardcoded_ips_folder, ip_file)

# individual parameters from personalize: bad for most tests as it may change due to slights changes in fit and/or personalize
from_personalize_ips_folder = os.path.join(test_data_dir, "individual_parameters", "from_personalize")
from_personalize_ip_path = lambda ip_file: os.path.join(from_personalize_ips_folder, ip_file)

# to store temporary data (used during tests)
test_tmp_dir = os.path.join(test_data_dir, "_tmp")
