import numpy as np
import torch
from scipy import stats
from ..attributes.attributes_logistic import Attributes_Logistic
from ..attributes.attributes_logistic_parallel import Attributes_LogisticParallel



def initialize_logistic_parallel(model, data, method="default"):

    # Dimension if not given
    model.dimension = data.dimension
    if model.source_dimension is None:
        model.source_dimension = int(np.sqrt(data.dimension))

    if method == "default":
        model.parameters = {
            'g': torch.tensor([1.]), 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': 0.1, 'deltas': torch.tensor([0.0] * (model.dimension - 1)),
            'betas': torch.zeros((model.dimension - 1, model.source_dimension))
        }
    elif method == "random":
        # Get the slopes / values / times mu and sigma
        slopes_mu, slopes_sigma = compute_patient_slopes_distribution(data)
        values_mu, values_sigma = compute_patient_values_distribution(data)
        time_mu, time_sigma = compute_patient_time_distribution(data)

        # Get random variations
        slopes = np.random.normal(loc=slopes_mu, scale=slopes_sigma)
        values = np.random.normal(loc=values_mu, scale=values_sigma)
        time = np.array(np.random.normal(loc=time_mu, scale=time_sigma))

        # Check that slopes are >0, values between 0 and 1
        slopes[slopes < 0] = 0.01
        values[values < 0] = 0.01
        values[values > 1] = 0.99

        # Do transformations
        t0 = torch.Tensor(time)
        v0_array = torch.Tensor(np.log((np.array(slopes))))
        g_array = torch.Tensor(np.exp(1 / (1 + values)))

        model.parameters = {
            'g': torch.tensor([torch.mean(g_array)]),
            'tau_mean': t0, 'tau_std': 2.0,
            'xi_mean': float(torch.mean(v0_array).detach().numpy()),'xi_std': 0.1,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': 0.1, 'deltas': torch.tensor([0.0] * (model.dimension - 1)),
            'betas': torch.zeros((model.dimension - 1, model.source_dimension))
        }

    else:
        raise ValueError("Initialization method not known")



    # Initialize the attribute
    model.attributes = Attributes_LogisticParallel(model.dimension, model.source_dimension)
    model.attributes.update(['all'], model.parameters) # TODO : why is this not needed ???
    model.is_initialized = True

    return model


def initialize_logistic(model, data, method="default"):

    # Dimension if not given
    model.dimension = data.dimension
    if model.source_dimension is None:
        model.source_dimension = int(np.sqrt(data.dimension))

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(data)
    values_mu, values_sigma = compute_patient_values_distribution(data)
    time_mu, time_sigma = compute_patient_time_distribution(data)

    # Method
    if method == "default":
        slopes = np.array(slopes_mu)
        values = np.array(values_mu)
        time = np.array(time_mu)
    elif method == "random":
        slopes = np.random.normal(loc=slopes_mu, scale=slopes_sigma)
        values = np.random.normal(loc=values_mu, scale=values_sigma)
        time = np.array(np.random.normal(loc=time_mu, scale=time_sigma))
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and 1
    slopes[slopes < 0] = 0.01
    values[values < 0] = 0.01
    values[values > 1] = 0.99

    # Do transformations
    t0 = torch.Tensor(time)
    v0_array = torch.Tensor(np.log((np.array(slopes))))
    g_array = torch.Tensor(np.exp(1/(1+values)))

    # Create smart initialization dictionnary
    SMART_INITIALIZATION = {
        'g': g_array,
        'v0': v0_array,
        'betas': torch.zeros((model.dimension - 1, model.source_dimension)),
        'tau_mean': t0, 'tau_std': 1.0,
        'xi_mean': .0, 'xi_std': 0.05,
        'sources_mean': 0.0, 'sources_std': 1.0,
        'noise_std': 0.1
    }

    # Initializes Parameters
    for parameter_key in model.parameters.keys():
        if model.parameters[parameter_key] is None:
            model.parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]

    # Initialize the attribute
    model.attributes = Attributes_Logistic(model.dimension, model.source_dimension)
    model.attributes.update(['all'], model.parameters)
    model.is_initialized = True

    return model


def compute_patient_slopes_distribution(data):
    """
    Linear Regression on each feature to get slopes
    :param data:
    :return:
    """

    # To Pandas
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    slopes_mu, slopes_sigma = [], []

    for dim in range(data.dimension):
        slope_dim_patients = []
        count = 0

        for idx in data.indices:
            # Select patient dataframe
            df_patient = df.loc[idx]  # .reset_index().set_index(['ID', 'TIME']) # TODO : Qu'est ce que ça doit faire?
            x = df_patient.index.get_level_values('TIME').values
            y = df_patient.iloc[:, dim].values

            # Delete if less than 2 visits
            if len(x) < 2:
                continue

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            slope_dim_patients.append(slope)
            count += 1

            # Stop at 50
            if count > 50:
                break

        slopes_mu.append(np.mean(slope_dim_patients))
        slopes_sigma.append(np.std(slope_dim_patients))

    return slopes_mu, slopes_sigma


def compute_patient_values_distribution(data):
    """
    Returns mu / sigma of given dataset values
    :param data:
    :return:
    """
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    return df.mean().values, df.std().values

def compute_patient_time_distribution(data):
    """
    Returns mu / sigma of given dataset times
    :param data:
    :return:
    """
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    return np.mean(df.index.get_level_values('TIME')), np.std(df.index.get_level_values('TIME'))