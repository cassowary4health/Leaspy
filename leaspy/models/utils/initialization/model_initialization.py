
import warnings

import torch
from scipy import stats
import numpy as np
#from sklearn.decomposition import PCA

xi_std = .5
tau_std = 5.
noise_std = .1
sources_std = 1.

def initialize_parameters(model, dataset, method="default"):
    """
    Initialize the model's group parameters given its name & the scores of all subjects.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters.
    """
    name = model.name
    if name in ['logistic', 'univariate_logistic']:
        parameters = initialize_logistic(model, dataset, method)
    elif name == 'logistic_parallel':
        parameters = initialize_logistic_parallel(model, dataset, method)
    elif name == 'logistic_asymptots':
        parameters = initialize_logistic_asymp(model, dataset, method)
    elif name == 'logistic_asymp_delay':
        parameters = initialize_logistic_asymp(model, dataset, method, delay = True)
    elif name == 'stannard':
        parameters = initialize_stannard(model, dataset, method)
    elif name in ['linear', 'univariate_linear']:
        parameters = initialize_linear(model, dataset, method)
    #elif name == 'univariate':
    #    parameters = initialize_univariate(dataset, method)
    elif name == 'mixed_linear-logistic':
        raise NotImplementedError
        #parameters = initialize_logistic(model, dataset, method)
    else:
        raise ValueError("There is no initialization method for the parameter of the model {}".format(name))

    return parameters


def initialize_logistic(model, dataset, method):
    """
    Initialize the logistic model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std' and 'noise_std'.
    """
    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)

    # Method
    if method == "default":
        slopes = slopes_mu
        values = values_mu
        time = time_mu
    elif method == "random":
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and 1
    slopes[slopes < 0] = 0.01
    values[values < 0] = 0.01
    values[values > 1] = 0.99

    # Do transformations
    t0 = time.clone()
    slopes = slopes.mean() * torch.ones_like(slopes)
    v0_array = slopes.log()
    g_array = torch.log(1. / values - 1.) # cf. Igor thesis; <!> exp is done in Attributes class for logisitic models
    betas = torch.zeros((model.dimension - 1, model.source_dimension))
    #normal = torch.distributions.normal.Normal(loc=0, scale=0.1)
    #betas = normal.sample(sample_shape=(model.dimension - 1, model.source_dimension))
    
    # Create smart initialization dictionary
    if 'univariate' in model.name:
        xi_mean = v0_array.squeeze() # already log'ed
        parameters = {
            'g': g_array.squeeze(),
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': xi_mean, 'xi_std': torch.tensor(xi_std),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': g_array,
            'v0': v0_array,
            'betas': betas,
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': torch.tensor(0.), 'xi_std': torch.tensor(xi_std),
            'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters

def initialize_logistic_parallel(model, dataset, method):
    """
    Initialize the logistic parallel model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g',  'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std', 'noise_std', 'delta' and 'beta'.
    """

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)

    if method == 'default':
        # Get random variations
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
        betas = torch.zeros((model.dimension - 1, model.source_dimension))
    elif method == 'random':
        # Get random variations
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
        betas = torch.distributions.normal.Normal.sample(sample_shape=(model.dimension - 1, model.source_dimension))
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and 1
    slopes[slopes < 0] = 0.01
    values[values < 0] = 0.01
    values[values > 1] = 0.99

    # Do transformations
    t0 = time.clone()
    v0_array = slopes.log()
    g_array = torch.log(1. / values - 1.) # cf. Igor thesis; <!> exp is done in Attributes class for logisitic models

    return {
        'g': torch.tensor([torch.mean(g_array)], dtype=torch.float32),
        'tau_mean': t0, 'tau_std': torch.tensor(tau_std, dtype=torch.float32),
        'xi_mean': torch.mean(v0_array).detach(), 'xi_std': torch.tensor(xi_std, dtype=torch.float32),
        'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
        'noise_std': torch.tensor([noise_std], dtype=torch.float32),
        'deltas': torch.tensor([0.0] * (model.dimension - 1), dtype=torch.float32),
        'betas': betas
    }



def initialize_stannard(model, dataset, method):
    """
    Initialize the logistic_asymp model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'Param=[infection,guerison]', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std' and 'noise_std'.
    """
    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)
    #We take the max values of each features to initialize the asymptots
   
    
    
    # Method
    if method == "default":
        slopes = slopes_mu
        values = values_mu
        time = time_mu
    elif method == "random":
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and rho the asymptots
   
    values[values < 0] = 0.01
    values[values >1] = 0.99
    
    
    # Do transformations
    t0 = time.clone()
    slopes = slopes.mean() * torch.ones_like(slopes)

    slopes = slopes / (values * (1-values))

    slopes = slopes.log()

    gammas = torch.zeros(model.dimension)

    g_array = torch.log(1./values - 1.) # every variable are stored in log forms
    betas = torch.zeros((model.dimension-1, model.source_dimension))
    #normal = torch.distributions.normal.Normal(loc=0, scale=0.1)
    #betas = normal.sample(sample_shape=(model.dimension, model.source_dimension))
    # param dim 1
    # Create smart initialization dictionary
    if 'univariate' in model.name:
        xi_mean = slopes.squeeze() # already log'ed
        parameters = {
            'g': g_array.squeeze(), 
            'v0': torch.tensor(slopes),
            'gammas': torch.tensor(gammas),
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': xi_mean, 'xi_std': torch.tensor(xi_std),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': g_array,
            'v0': torch.tensor(slopes),
            'gammas': torch.tensor(gammas),
            'betas': betas,
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': torch.tensor(0.), 'xi_std': torch.tensor(xi_std),
            'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters


def initialize_logistic_asymp(model, dataset, method, delay=False):
    """
    Initialize the logistic_asymp model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'Param=[infection,guerison]', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std' and 'noise_std'.
    """
    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)
    #We take the max values of each features to initialize the asymptots
   
    asymptots = compute_features_quantile_values(dataset)
    asymptots = torch.clamp(asymptots, min = 0, max = model.max_asymptot - 0.05)
    
    # Method
    if method == "default":
        slopes = slopes_mu
        values = values_mu
        time = time_mu
    elif method == "random":
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and rho the asymptots
   
    values[values < 0] = 0.01

    values[values > asymptots] = asymptots[values > asymptots] - 0.05
    # Do transformations
    t0 = time.clone()
    print("slopes mean")
    print(slopes)
    slopes = slopes.mean() * torch.ones_like(slopes)

    slopes = slopes / (values * (asymptots - values))
    
 
    asymptots = asymptots.log()#On sample sur rho plutôt que sur guerison
    slopes = slopes.log()

    g_array = torch.log(asymptots / values - 1.) # every variable are stored in log forms
    betas = torch.zeros((model.dimension, model.source_dimension))
    #normal = torch.distributions.normal.Normal(loc=0, scale=0.1)
    #betas = normal.sample(sample_shape=(model.dimension, model.source_dimension))
    # param dim 1
    # Create smart initialization dictionary
    if 'univariate' in model.name:
        xi_mean = slopes.squeeze() # already log'ed
        parameters = {
            'g': g_array.squeeze(), 
            'v0': torch.tensor(slopes),
            'asymptots': torch.tensor(asymptots),
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': xi_mean, 'xi_std': torch.tensor(xi_std),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': g_array,
            'v0': torch.tensor(slopes),
            'asymptots': torch.tensor(asymptots),
            'betas': betas,
            'tau_mean': t0, 'tau_std': torch.tensor(tau_std),
            'xi_mean': torch.tensor(0.), 'xi_std': torch.tensor(xi_std),
            'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }
        if delay:
            parameters['betas_asymptots'] = torch.zeros((model.dimension, model.source_dimension))

    return parameters

def initialize_logistic_parallel(model, dataset, method):
    """
    Initialize the logistic parallel model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.
    method: `str`
        Must be one of:
        - "default": initialize at mean.
        - "random":  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g',  'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std', 'noise_std', 'delta' and 'beta'.
    """

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)

    if method == 'default':
        # Get random variations
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
        betas = torch.zeros((model.dimension - 1, model.source_dimension))
    elif method == 'random':
        # Get random variations
        slopes = torch.normal(slopes_mu, slopes_sigma)
        values = torch.normal(values_mu, values_sigma)
        time = torch.normal(time_mu, time_sigma)
        betas = torch.distributions.normal.Normal.sample(sample_shape=(model.dimension - 1, model.source_dimension))
    else:
        raise ValueError("Initialization method not known")

    # Check that slopes are >0, values between 0 and 1
    slopes[slopes < 0] = 0.01
    values[values < 0] = 0.01
    values[values > 1] = 0.99

    # Do transformations
    t0 = time.clone()
    v0_array = slopes.log()
    g_array = torch.log(1. / values - 1.) # cf. Igor thesis; <!> exp is done in Attributes class for logisitic models

    return {
        'g': torch.tensor([torch.mean(g_array)], dtype=torch.float32),
        'tau_mean': t0, 'tau_std': torch.tensor(tau_std, dtype=torch.float32),
        'xi_mean': torch.mean(v0_array).detach(), 'xi_std': torch.tensor(xi_std, dtype=torch.float32),
        'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
        'noise_std': torch.tensor([noise_std], dtype=torch.float32),
        'deltas': torch.tensor([0.0] * (model.dimension - 1), dtype=torch.float32),
        'betas': betas
    }


def initialize_linear(model, dataset, method):
    """
    Initialize the linear model's group parameters.

    Parameters
    ----------
    model: a leaspy model class object
        The model to initialize.
    dataset: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores.

    Returns
    -------
    parameters: `dict` [`str`, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std' and 'noise_std'.
    """
    sum_ages = torch.sum(dataset.timepoints).item()
    nb_nonzeros = (dataset.timepoints != 0).sum()

    t0 = float(sum_ages) / float(nb_nonzeros)

    df = dataset.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)

    positions, velocities = [[] for _ in range(model.dimension)], [[] for _ in range(model.dimension)]

    for idx in dataset.indices:
        indiv_df = df.loc[idx]
        ages = indiv_df.index.values
        features = indiv_df.values

        if len(ages) == 1:
            continue

        for dim in range(model.dimension):

            ages_list, feature_list = [], []
            for i, f in enumerate(features[:, dim]):
                if f == f:
                    feature_list.append(f)
                    ages_list.append(ages[i])

            if len(ages_list) < 2:
                break
            else:
                slope, intercept, _, _, _ = stats.linregress(ages_list, feature_list)

                value = intercept + t0 * slope

                velocities[dim].append(slope)
                positions[dim].append(value)

    positions = [torch.tensor(_) for _ in positions]
    positions = torch.tensor([torch.mean(_) for _ in positions], dtype=torch.float32)
    velocities = [torch.tensor(_) for _ in velocities]
    velocities = torch.tensor([torch.mean(_) for _ in velocities], dtype=torch.float32)

    if 'univariate' in model.name:
        if (velocities <= 0).item():
            warnings.warn("Individual linear regressions made at initialization has a mean slope which is negative: not properly handled in case of an univariate linear model...")
            xi_mean = torch.tensor(-3.) # default...
        else:
            xi_mean = torch.log(velocities).squeeze()

        parameters = {
            'g': positions.squeeze(),
            'tau_mean': torch.tensor(t0), 'tau_std': torch.tensor(tau_std),
            'xi_mean': xi_mean, 'xi_std': torch.tensor(xi_std),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': positions,
            'v0': velocities,
            'betas': torch.zeros((model.dimension - 1, model.source_dimension)),
            'tau_mean': torch.tensor(t0), 'tau_std': torch.tensor(tau_std),
            'xi_mean': torch.tensor(0.), 'xi_std': torch.tensor(xi_std),
            'sources_mean': torch.tensor(0.), 'sources_std': torch.tensor(sources_std),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters

#def initialize_univariate(dataset, method):
#    # TODO?
#    return 0


def compute_patient_slopes_distribution(data):
    """
    Linear Regression on each feature to get slopes

    :param data: leaspy.io.data.dataset class object
    :return: slopes_mu : list of floats, slopes_sigma : list of floats
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
            df_patient = df.loc[idx]
            df_patient_dim = df_patient.iloc[:, dim].dropna()
            x = df_patient_dim.index.get_level_values('TIME').values
            y = df_patient_dim.values

            # Delete if less than 2 visits
            if len(x) < 2:
                continue
            # TODO : DO something if everyone has less than 2 visits

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            slope_dim_patients.append(slope)
            count += 1

            # Stop at 50
            #if count > 50:
                #break

        slopes_mu.append(torch.mean(torch.tensor(slope_dim_patients)).item())
        slopes_sigma.append(torch.mean(torch.tensor(slope_dim_patients)).item())

    return torch.tensor(slopes_mu), torch.tensor(slopes_sigma)


def compute_patient_values_distribution(data):
    """
    Returns means and standard deviations for the features of the given dataset values.

    Parameters
    ----------
    data: a leaspy.io.data.dataset.Dataset class object
        Contains the scores of all the subjects.

    Returns
    -------
    - means: `torch.Tensor`
        One mean per feature.
    - std: `torch.Tensor`
        One standard deviation per feature.
    """
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    return torch.tensor(df.mean().values, dtype=torch.float32), torch.tensor(df.std().values, dtype=torch.float32)
def compute_features_max_values(data):
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    MAX=[]

    for dim in range(data.dimension):
        m=0
        count = 0

        for idx in data.indices:
            # Select patient dataframe
            df_patient = df.loc[idx]
            df_patient_dim = df_patient.iloc[:, dim].dropna()
            m=max(m,df_patient_dim.max())
        MAX.append(m)

    return torch.tensor(MAX, dtype=torch.float32)

def compute_features_quantile_values(data):
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    Quant=[]

    for dim in range(data.dimension):
        m=0
        count = 0
        L=[]
        for idx in data.indices:
            # Select patient dataframe
            df_patient = df.loc[idx]
            df_patient_dim = df_patient.iloc[:, dim].dropna()
            L=L+df_patient_dim.tolist()
        m=np.quantile(np.array(L),0.96)
        Quant.append(m)

    return torch.tensor(Quant, dtype=torch.float32)


def compute_patient_time_distribution(data):
    """
    Returns mu / sigma of given dataset times.

    Parameters
    ----------
    data: a leaspy.io.data.dataset.Dataset class object
        Contains the individual scores

    Returns
    -------
    - mean: `torch.Tensor`
    - sigma: `torch.Tensor`
    """
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    return torch.mean(torch.tensor(df.index.get_level_values('TIME').tolist())), \
           torch.std(torch.tensor(df.index.get_level_values('TIME').tolist()))


'''
def initialize_logistic_parallel(model, data, method="default"):

    # Dimension if not given
    model.dimension = data.dimension
    if model.source_dimension is None:
        model.source_dimension = int(np.sqrt(data.dimension))

    if method == "default":
        model.parameters = {
            'g': torch.tensor([1.]), 'tau_mean': 70.0, 'tau_std': 2.0, 'xi_mean': -3., 'xi_std': 0.1,
            'sources_mean': 0.0, 'sources_std': 1.0,
            'noise_std': torch.tensor([0.1], dtype=torch.float32),
            'deltas': torch.tensor([0.0] * (model.dimension - 1)),
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
            'noise_std': torch.tensor([0.1], dtype=torch.float32),
            'deltas': torch.tensor([0.0] * (model.dimension - 1)),
            'betas': torch.zeros((model.dimension - 1, model.source_dimension))
        }

    else:
        raise ValueError("Initialization method not known")

    # Initialize the attribute
    model.attributes = AttributesLogisticParallel(model.dimension, model.source_dimension)
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
        'noise_std': torch.tensor([0.1], dtype=torch.float32)
    }

    # Initializes Parameters
    for parameter_key in model.parameters.keys():
        if model.parameters[parameter_key] is None:
            model.parameters[parameter_key] = SMART_INITIALIZATION[parameter_key]
        else:
            print('ok')

    # Initialize the attribute
    model.attributes = AttributesLogistic(model.dimension, model.source_dimension)
    model.attributes.update(['all'], model.parameters)
    model.is_initialized = True

    return model
'''

