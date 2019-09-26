import timeit


def test_fit_logistic_small():
    from leaspy import Leaspy, Data, AlgorithmSettings
    from tests import example_data_path

    # Inputs
    data = Data.from_csv_file(example_data_path)
    algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

    # Initialize
    leaspy = Leaspy("logistic")
    leaspy.model.load_hyperparameters({'source_dimension': 2})

    # Fit the model on the data
    leaspy.fit(data, algorithm_settings=algo_settings)


def test_fit_logistic_big_setup(n_patients, n_visits_per_patient, n_modalities):
    from leaspy import Data
    import pandas as pd
    import numpy as np
    import time

    start = time.time()

    # n_patients = 200
    # n_visits_per_patient = 10
    # n_modalities = 8

    df = pd.DataFrame()

    # Inputs
    for i in range(n_patients):
        patient = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, n_modalities))
        times = np.random.uniform(low=0.01, high=0.99, size=(n_visits_per_patient, 1))
        patient_df = pd.DataFrame(patient)
        patient_df.columns = [str(col) for col in patient_df.columns]
        patient_df['ID'] = i
        patient_df['TIME'] = times
        df = pd.concat([df, patient_df])

    df = df[["ID", "TIME"] + [str(i) for i in range(n_modalities)]]

    global data
    data = Data.from_dataframe(df)

    print("test_fit_logistic_big_setup execution time: %.2f s" % (time.time() - start))


def test_fit_logistic_big():
    from leaspy import Leaspy, AlgorithmSettings

    algo_settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

    # Initialize
    leaspy = Leaspy("logistic")
    leaspy.model.load_hyperparameters({'source_dimension': 2})

    # Fit the model on the data
    leaspy.fit(data, algorithm_settings=algo_settings)


if __name__ == '__main__':
    import numpy as np

    print("start")

    it = 2

    # t = timeit.timeit(test_fit_logistic_small, number=it)
    # t = timeit.timeit(test_fit_logistic_big, setup=test_fit_logistic_big_setup, number=it)
    # print(t/it)

    t = timeit.repeat(test_fit_logistic_big,
                      setup=lambda: test_fit_logistic_big_setup(n_patients=200, n_visits_per_patient=10, n_modalities=8),
                      number=it, repeat=3)

    print([i / it for i in t])
    print(np.mean([i / it for i in t]))

    # for n_patients in [20, 100, 500, 1000, 1500, 2000]:
    #     for n_modalities in [2, 4, 8, 16, 32]:
    #         print("n_patients=%d, n_modalities=%d" % (n_patients, n_modalities))
    #         t = timeit.repeat(test_fit_logistic_big,
    #                           setup=lambda: test_fit_logistic_big_setup(n_patients=n_patients, n_visits_per_patient=10, n_modalities=n_modalities),
    #                           number=it, repeat=3)
    #
    #         print(np.mean([i/it for i in t]))
