class SimulateParameter:
    def __init__(self, kernel, ss, df_mean, df_cov, noise_generator, headers, get_sources, sources_method, density_type, mean_num_visits, std_num_visits, tau_mean, mean_age, std_age):
        self.kernel = kernel
        self.ss = ss
        self.df_mean = df_mean
        self.df_cov = df_cov
        self.noise_generator = noise_generator
        self.headers = headers
        self.get_sources = get_sources
        self.sources_method = sources_method
        self.density_type = density_type
        self.mean_num_visits = mean_num_visits
        self.std_num_visits = std_num_visits
        self.tau_mean = tau_mean
        self.mean_age = mean_age
        self.std_age = std_age

