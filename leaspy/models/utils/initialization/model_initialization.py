import warnings

import torch
from numpy import exp
from scipy import stats

# <!> circular imports
import leaspy

#from joblib import Parallel, delayed

xi_std = .5
tau_std = 5.
noise_std = .1
sources_std = 1.

def initialize_parameters(model, dataset, method="default"):
    """
    Initialize the model's group parameters given its name & the scores of all subjects.

    Under-the-hood it calls an initialization function dedicated for the `model`:
        * :func:`.initialize_linear` (including when `univariate`)
        * :func:`.initialize_logistic` (including when `univariate`)
        * :func:`.initialize_logistic_parallel`

    It is automatically called during :meth:`.Leaspy.fit`.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    dataset : :class:`.Dataset`
        Contains the individual scores.
    method: str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: dict [str, :class:`torch.Tensor`]
        Contains the initialized model's group parameters.
    """

    if method == 'lme':
        return lme_init(model, dataset) # support kwargs?

    name = model.name
    if name in ['logistic', 'univariate_logistic']:
        parameters = initialize_logistic(model, dataset, method)
    elif name == 'logistic_parallel':
        parameters = initialize_logistic_parallel(model, dataset, method)
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


def get_lme_results(dataset, n_jobs=-1, **lme_fit_kwargs):
    """
    Fit a LME on univariate (per feature) time-series (feature vs. patients' ages with varying intercept & slope)

    Parameters
    ----------
    dataset: leaspy.io.data.dataset.Dataset
        Contains all the data wrapped into a leaspy Dataset object
    n_jobs: int
        Number of jobs in parallel when multiple features to init
        Not used, buggy
    **lme_fit_kwargs:
        Other kwargs passed to 'lme_fit' (such as `force_independent_random_effects`, default True)

    Returns
    -------
    dict:
        {param: str -> param_values_for_ft: torch.Tensor(nb_fts, *shape_param)}
    """

    lme_fit_kwargs = {
        'with_random_slope_age': True,
        'force_independent_random_effects': True,
        **lme_fit_kwargs} # defaults

    #@delayed
    def fit_one_ft(df_ft):
        data_ft = leaspy.Data.from_dataframe(df_ft)
        lsp_lme_ft = leaspy.Leaspy('lme')
        algo = leaspy.AlgorithmSettings('lme_fit', **lme_fit_kwargs) # seed=seed

        lsp_lme_ft.fit(data_ft, algo)

        return lsp_lme_ft.model.parameters

    df = dataset.to_pandas().set_index(['ID','TIME'])
    #res = Parallel(n_jobs=n_jobs)(delayed(fit_one_ft)(df.loc[:, [ft]].dropna().copy()) for ft in dataset.headers)
    res = list(fit_one_ft(df.loc[:, [ft]].dropna()) for ft in dataset.headers)

    # output a dict of tensor stacked by feature, indexed by param
    param_names = next(iter(res)).keys()

    return {
        param_name: torch.stack([torch.tensor(res_ft[param_name], dtype=torch.float32)
                                 for res_ft in res])
        for param_name in param_names
    }

def lme_init(model, dataset, fact_std=1., **kwargs):

    name = model.name
    loss = model.loss # has to be set directly at model init and not in algosettings step to be available here
    assert dataset.headers == model.features

    multiv = 'univariate' not in name

    print('Initialization with linear mixed-effects model...')
    lme = get_lme_results(dataset, **kwargs)
    print()

    # init
    params = {}

    v0_lin = (lme['fe_params'][:, 1] / lme['ages_std']).clamp(min=1e-2) # > exp(-4.6)

    if 'linear' in name:
        # global tau mean (arithmetic mean of ages mean)
        params['tau_mean'] = lme['ages_mean'].mean()

        params['g'] = lme['fe_params'][:, 0] + v0_lin * (params['tau_mean'] - lme['ages_mean'])
        params['v0' if multiv else 'xi_mean'] = v0_lin.log()

    #elif name in ['logistic_parallel']:
    #    # deltas = torch.zeros((model.dimension - 1,), dtype=torch.float32) ?
    #    pass # TODO...
    elif name in ['logistic', 'univariate_logistic']:

        """
        # global tau mean (arithmetic mean of inflexion point per feature)
        t0_ft = lme['ages_mean'] + (.5 - lme['fe_params'][:, 0]) / v0_lin # inflexion pt
        params['tau_mean'] = t0_ft.mean()
        """

        # global tau mean (arithmetic mean of ages mean)
        params['tau_mean'] = lme['ages_mean'].mean()

        # positions at this tau mean
        pos_ft = lme['fe_params'][:, 0] + v0_lin * (params['tau_mean'] - lme['ages_mean'])

        # parameters under leaspy logistic formulation
        g = 1/pos_ft.clamp(min=1e-2, max=1-1e-2) - 1
        params['g'] = g.log() # -4.6 ... 4.6

        v0 = g/(1+g)**2 * 4 * v0_lin # slope from lme at inflexion point

        #if name == 'logistic_parallel':
        #    # a common speed for all features!
        #    params['xi_mean'] = v0.log().mean() # or log of fts-mean?
        #else:
        params['v0' if multiv else 'xi_mean'] = v0.log()

    else:
        raise ValueError(f"Model {name} is not supported in `lme` initialization.")

    ## Dispersion of individual parameters
    # approx. dispersion on tau (-> on inflexion point when logistic)
    tau_var_ft = lme['cov_re'][:, 0,0] / v0_lin ** 2
    params['tau_std'] = fact_std * (1/tau_var_ft).mean() ** -.5  # harmonic mean on variances per ft

    # approx dispersion on alpha and then xi
    alpha_var_ft = lme['cov_re'][:, 1,1] / lme['fe_params'][:, 1]**2
    xi_var_ft = (1/2+(1/4+alpha_var_ft)**.5).log() # because alpha = exp(xi) so var(alpha) = exp(2*var_xi) - exp(var_xi)
    params['xi_std'] = fact_std * (1/xi_var_ft).mean() ** -.5

    # Residual gaussian noise
    if 'diag_noise' in loss:
        params['noise_std'] = fact_std * lme['noise_std']
    else:
        # arithmetic mean on variances
        params['noise_std'] = fact_std * (lme['noise_std'] ** 2).mean().reshape((1,)) ** .5 # 1D tensor

    # For multivariate models, xi_mean == 0.
    if name in ['linear', 'logistic']: # isinstance(model, MultivariateModel)
        params['xi_mean'] = torch.tensor(0., dtype=torch.float32)

    if multiv: # including logistic_parallel
        params['betas'] = torch.zeros((model.dimension - 1, model.source_dimension), dtype=torch.float32)
        params['sources_mean'] = torch.tensor(0., dtype=torch.float32)
        params['sources_std'] = torch.tensor(sources_std, dtype=torch.float32)

    return params


def initialize_logistic(model, dataset, method):
    """
    Initialize the logistic model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    dataset : :class:`.Dataset`
        Contains the individual scores.
    method: str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters.
        The parameters' keys are 'g', 'v0', 'betas', 'tau_mean',
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
    slopes = slopes.clamp(min=1e-2)
    values = values.clamp(min=1e-2, max=1-1e-2)

    # Do transformations
    t0 = time.clone().detach()
    v0_array = slopes.log().detach()
    g_array = torch.log(1. / values - 1.).detach() # cf. Igor thesis; <!> exp is done in Attributes class for logisitic models
    betas = torch.zeros((model.dimension - 1, model.source_dimension))
    # normal = torch.distributions.normal.Normal(loc=0, scale=0.1)
    # betas = normal.sample(sample_shape=(model.dimension - 1, model.source_dimension))

    # Create smart initialization dictionary
    if 'univariate' in model.name:
        xi_mean = v0_array.squeeze() # already log'ed
        parameters = {
            'g': g_array.squeeze(),
            'tau_mean': t0,
            'tau_std': torch.tensor(tau_std, dtype=torch.float32),
            'xi_mean': xi_mean,
            'xi_std': torch.tensor(xi_std, dtype=torch.float32),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': g_array,
            'v0': v0_array,
            'betas': betas,
            'tau_mean': t0,
            'tau_std': torch.tensor(tau_std, dtype=torch.float32),
            'xi_mean': torch.tensor(0., dtype=torch.float32),
            'xi_std': torch.tensor(xi_std, dtype=torch.float32),
            'sources_mean': torch.tensor(0., dtype=torch.float32),
            'sources_std': torch.tensor(sources_std, dtype=torch.float32),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters


def initialize_logistic_parallel(model, dataset, method):
    """
    Initialize the logistic parallel model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    dataset : :class:`.Dataset`
        Contains the individual scores.
    method: str
        Must be one of:
            * ``'default'``: initialize at mean.
            * ``'random'``:  initialize with a gaussian realization with same mean and variance.

    Returns
    -------
    parameters: dict [str, `torch.Tensor`]
        Contains the initialized model's group parameters. The parameters' keys are 'g',  'tau_mean',
        'tau_std', 'xi_mean', 'xi_std', 'sources_mean', 'sources_std', 'noise_std', 'delta' and 'beta'.
    """

    # Get the slopes / values / times mu and sigma
    slopes_mu, slopes_sigma = compute_patient_slopes_distribution(dataset)
    values_mu, values_sigma = compute_patient_values_distribution(dataset)
    time_mu, time_sigma = compute_patient_time_distribution(dataset)

    if method == 'default':
        slopes = slopes_mu
        values = values_mu
        time = time_mu
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
    slopes = slopes.clamp(min=1e-2)
    values = values.clamp(min=1e-2, max=1-1e-2)

    # Do transformations
    t0 = time.clone()
    v0 = slopes.log().mean().detach()
    #v0 = slopes.mean().log().detach() # mean before log
    g = torch.log(1. / values - 1.).mean().detach() # cf. Igor thesis; <!> exp is done in Attributes class for logisitic models
    #g = torch.log(1. / values.mean() - 1.).detach() # mean before transfo

    return {
        'g': g,
        'tau_mean': t0,
        'tau_std': torch.tensor(tau_std, dtype=torch.float32),
        'xi_mean': v0,
        'xi_std': torch.tensor(xi_std, dtype=torch.float32),
        'sources_mean': torch.tensor(0., dtype=torch.float32),
        'sources_std': torch.tensor(sources_std, dtype=torch.float32),
        'noise_std': torch.tensor([noise_std], dtype=torch.float32),
        'deltas': torch.zeros((model.dimension - 1,), dtype=torch.float32),
        'betas': betas
    }


def initialize_linear(model, dataset, method):
    """
    Initialize the linear model's group parameters.

    Parameters
    ----------
    model : :class:`.AbstractModel`
        The model to initialize.
    dataset : :class:`.Dataset`
        Contains the individual scores.

    Returns
    -------
    parameters: dict [str, `torch.Tensor`]
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

    neg_velocities = velocities <= 0
    if neg_velocities.any():
        warnings.warn(f"Mean slope of individual linear regressions made at initialization is negative for {[f for f, vel in zip(model.features, velocities) if vel <= 0]}: not properly handled in model...")
    velocities = velocities.clamp(min=1e-2)

    # always take the log (even in non univariate model!)
    velocities = torch.log(velocities).detach()

    if 'univariate' in model.name:
        xi_mean = velocities.squeeze()

        parameters = {
            'g': positions.squeeze(),
            'tau_mean': torch.tensor(t0, dtype=torch.float32),
            'tau_std': torch.tensor(tau_std, dtype=torch.float32),
            'xi_mean': xi_mean,
            'xi_std': torch.tensor(xi_std, dtype=torch.float32),
            'noise_std': torch.tensor(noise_std, dtype=torch.float32)
        }
    else:
        parameters = {
            'g': positions,
            'v0': velocities,
            'betas': torch.zeros((model.dimension - 1, model.source_dimension)),
            'tau_mean': torch.tensor(t0, dtype=torch.float32),
            'tau_std': torch.tensor(tau_std, dtype=torch.float32),
            'xi_mean': torch.tensor(0., dtype=torch.float32),
            'xi_std': torch.tensor(xi_std, dtype=torch.float32),
            'sources_mean': torch.tensor(0., dtype=torch.float32),
            'sources_std': torch.tensor(sources_std, dtype=torch.float32),
            'noise_std': torch.tensor([noise_std], dtype=torch.float32)
        }

    return parameters


#def initialize_univariate(dataset, method):
#    # TODO?
#    return 0


def compute_patient_slopes_distribution(data, max_inds=None):
    """
    Linear Regression on each feature to get slopes

    Parameters
    ----------
    data : :class:`.Dataset`

    Returns
    -------
    slopes_mu : :class:`torch.Tensor` [n_features,]
    slopes_sigma : :class:`torch.Tensor` [n_features,]
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

            # Stop at `max_inds`
            if max_inds and count > max_inds:
                break

        slopes_mu.append(torch.mean(torch.tensor(slope_dim_patients)).item())
        slopes_sigma.append(torch.std(torch.tensor(slope_dim_patients)).item())

    return torch.tensor(slopes_mu), torch.tensor(slopes_sigma)


def compute_patient_values_distribution(data):
    """
    Returns means and standard deviations for the features of the given dataset values.

    Parameters
    ----------
    data : :class:`.Dataset`
        Contains the scores of all the subjects.

    Returns
    -------
    means : :class:`torch.Tensor` [n_features,]
        One mean per feature.
    std : :class:`torch.Tensor` [n_features,]
        One standard deviation per feature.
    """
    df = data.to_pandas()
    df.set_index(["ID", "TIME"], inplace=True)
    return torch.tensor(df.mean().values, dtype=torch.float32), torch.tensor(df.std().values, dtype=torch.float32)


def compute_patient_time_distribution(data):
    """
    Returns mu / sigma of given dataset times.

    Parameters
    ----------
    data : :class:`.Dataset`
        Contains the individual scores

    Returns
    -------
    mean : :class:`torch.Tensor` scalar
    sigma : :class:`torch.Tensor` scalar
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
        slopes = slopes.clamp(min=1e-2)
        values = values.clamp(min=1e-2, max=1-1e-2)

        # Do transformations
        t0 = torch.tensor(time)
        v0_array = torch.tensor(np.log((np.array(slopes))))
        g_array = torch.tensor(np.exp(1 / (1 + values)))

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
    model.attributes = LogisticParallelAttributes(model.dimension, model.source_dimension)
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
    slopes = slopes.clamp(min=1e-2)
    values = values.clamp(min=1e-2, max=1-1e-2)

    # Do transformations
    t0 = torch.tensor(time)
    v0_array = torch.tensor(np.log((np.array(slopes))))
    g_array = torch.tensor(np.exp(1/(1+values)))

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
    model.attributes = LogisticAttributes(model.dimension, model.source_dimension)
    model.attributes.update(['all'], model.parameters)
    model.is_initialized = True

    return model
'''
