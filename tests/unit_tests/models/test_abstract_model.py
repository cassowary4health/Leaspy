import unittest
import os

import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.models.abstract_model import AbstractModel
from leaspy.models.model_factory import ModelFactory

from tests import example_data_path, binary_data_path, hardcoded_model_path
from tests.helpers import TestHelpers


class AbstractModelTest(unittest.TestCase):

    @TestHelpers.allow_abstract_class_init(AbstractModel)
    def test_abstract_model_constructor(self):
        """
        Test initialization of abstract model class object.
        """
        model = AbstractModel("dummy_abstractmodel")
        self.assertFalse(model.is_initialized)
        self.assertEqual(model.name, "dummy_abstractmodel")
        self.assertEqual(model.parameters, None)
        self.assertIsInstance(model.distribution, torch.distributions.normal.Normal)

        # Test the presence of all these essential methods
        main_methods = ['load_parameters', 'get_individual_variable_name', 'compute_sum_squared_tensorized',
                        'compute_individual_attachment_tensorized_mcmc', 'compute_individual_attachment_tensorized',
                        'update_model_parameters', 'update_model_parameters_burn_in',
                        'get_population_realization_names', 'get_individual_realization_names',
                        'compute_regularity_realization', 'compute_regularity_variable', 'get_realization_object',
                        'compute_individual_ages_from_biomarker_values',
                        'compute_individual_ages_from_biomarker_values_tensorized']

        present_attributes = [_ for _ in dir(model) if _[:2] != '__']  # Get the present method

        for attribute in main_methods:
            self.assertIn(attribute, present_attributes)
        # TODO: use python's hasattr and issubclass

    @TestHelpers.allow_abstract_class_init(AbstractModel)
    def test_load_parameters(self):
        """
        Test the method load_parameters.
        """
        leaspy_object = Leaspy.load(hardcoded_model_path('logistic_scalar_noise'))

        abstract_model = AbstractModel("dummy_model")

        abstract_model.load_parameters(leaspy_object.model.parameters)

        self.assertTrue(torch.equal(abstract_model.parameters['g'],
                                    torch.tensor([0.5, 1.5, 1.0, 2.0])))
        self.assertTrue(torch.equal(abstract_model.parameters['v0'],
                                    torch.tensor([-2.0, -3.5, -3.0, -2.5])))
        self.assertTrue(torch.equal(abstract_model.parameters['betas'],
                                    torch.tensor([[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]])))
        self.assertTrue(torch.equal(abstract_model.parameters['tau_mean'], torch.tensor(75.2)))
        self.assertTrue(torch.equal(abstract_model.parameters['tau_std'], torch.tensor(7.1)))
        self.assertTrue(torch.equal(abstract_model.parameters['xi_mean'], torch.tensor(0.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['xi_std'], torch.tensor(0.2)))
        self.assertTrue(torch.equal(abstract_model.parameters['sources_mean'], torch.tensor(0.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['sources_std'], torch.tensor(1.0)))
        self.assertTrue(torch.equal(abstract_model.parameters['noise_std'], torch.tensor(0.2)))

    def test_all_model_run(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):
            leaspy = Leaspy(model_name)
            settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

            df = pd.read_csv(example_data_path)
            if model_name == 'univariate_linear' or model_name == 'univariate_logistic':
                df = df.iloc[:, :3]
            data = Data.from_dataframe(df)

            leaspy.fit(data, settings)

            methods = ['mode_real', 'mean_real', 'scipy_minimize']
            #if model_name not in ['logistic', 'logistic_parallel']:
            #    # problem with nans with 'gradient_descent_personalize' in multivariate logistic models
            #    methods.append('gradient_descent_personalize')

            for method in methods:
                burn_in_kw = dict() # not for all algos
                if '_real' in method:
                    burn_in_kw = dict(n_burn_in_iter=90, )
                settings = AlgorithmSettings(method, n_iter=100, seed=0, **burn_in_kw)
                result = leaspy.personalize(data, settings)

    def test_all_model_run_crossentropy(self):
        """
        Check if the following models run with the following algorithms.
        """
        for model_name in ('linear', 'univariate_logistic', 'univariate_linear', 'logistic', 'logistic_parallel'):
            leaspy = Leaspy(model_name, loss="crossentropy")
            settings = AlgorithmSettings('mcmc_saem', n_iter=200, seed=0)

            df = pd.read_csv(binary_data_path)
            if model_name == 'univariate_linear' or model_name == 'univariate_logistic':
                df = df.iloc[:, :3]
            data = Data.from_dataframe(df)

            leaspy.fit(data, settings)

            for method in ['scipy_minimize']:
                burn_in_kw = dict() # not for all algos
                if '_real' in method:
                    burn_in_kw = dict(n_burn_in_iter=90, )
                settings = AlgorithmSettings(method, n_iter=100, seed=0, **burn_in_kw)
                result = leaspy.personalize(data, settings)

    def test_tensorize_2D(self):

        t5 = torch.tensor([[5]],dtype=torch.float32)

        for x, unsqueeze_dim, expected_out in zip([
            [1,2], [1,2], 5, 5, [5], [5]
        ], [0,-1,0,-1,0,-1], [
            torch.tensor([[1,2]],dtype=torch.float32),
            torch.tensor([[1],[2]],dtype=torch.float32),
            t5, t5, t5, t5
        ]):
            self.assertTrue(torch.equal(
                AbstractModel._tensorize_2D(x,unsqueeze_dim=unsqueeze_dim),
                expected_out
            ))

    def test_audit_individual_parameters(self):

        # tuple: (valid,nb_inds,src_dim), ips_as_dict
        all_ips = [
            # 0 individual
            ((True, 0, 0), {'tau':[],'xi':[]}),
            ((True, 0, 5), {'tau':[],'xi':[],'sources':[]}), # src_dim undefined here...

            # 1 individual
            ((True, 1, 0), {'tau':50,'xi':0,}),
            ((False, 1, 1), {'tau':50,'xi':0,'sources':0}), # faulty (source should be vector)
            ((True, 1, 1), {'tau':50,'xi':0,'sources':[0]}),
            ((True, 1, 2), {'tau':50,'xi':0,'sources':[0,0]}),

            # 2 individuals
            ((True, 2, 0), {'tau':[50,60],'xi':[0,0.1],}),
            ((True, 2, 1), {'tau':[50,60],'xi':[0,0.1],'sources':[0,0.1]}), # accepted even if ambiguous
            ((True, 2, 1), {'tau':[50,60],'xi':[0,0.1],'sources':[[0],[0.1]]}), # cleaner
            ((True, 2, 2), {'tau':[50,60],'xi':[0,0.1],'sources':[[0,-1],[0.1,0]]}),

            # Faulty
            ((False, 1, 0), {'tau':0,'xi':0,'extra':0}),
            ((False, 1, 0), {'tau':0,}),
            ((False, None, 0), {'tau':[50,60],'xi':[0]}),
        ]

        for src_compat, m in [
            (lambda src_dim: src_dim <= 0, ModelFactory.model('univariate_logistic')),
            (lambda src_dim: src_dim >= 0, ModelFactory.model('logistic'))
        ]:

            for (valid,n_inds,src_dim), ips in all_ips:

                if m.name == 'logistic':
                    m.source_dimension = src_dim

                if (not valid) or (not src_compat(src_dim)):
                    with self.assertRaises(ValueError, ):
                        ips_info = m._audit_individual_parameters(ips)
                    continue

                ips_info = m._audit_individual_parameters(ips)

                keys = set(ips_info.keys()).symmetric_difference({'nb_inds','tensorized_ips','tensorized_ips_gen'})
                self.assertEqual(len(keys), 0)

                self.assertEqual(ips_info['nb_inds'], n_inds)

                list_t_ips = list(ips_info['tensorized_ips_gen'])
                self.assertEqual(len(list_t_ips), n_inds)

                t_ips = ips_info['tensorized_ips']
                self.assertIsInstance(t_ips, dict)
                keys_ips = set(t_ips.keys()).symmetric_difference(ips.keys())
                self.assertEqual(len(keys_ips), 0)

                for k,v in t_ips.items():
                    self.assertIsInstance(v, torch.Tensor)
                    self.assertEqual(v.dim(), 2)
                    self.assertEqual(v.shape, (n_inds, src_dim if (k == 'sources') and (n_inds > 0) else 1))

                if n_inds == 1:
                    t_ips0 = list_t_ips[0]
                    self.assertTrue(all(torch.equal(t_ips0[k], v) for k,v in t_ips.items())) # because only 1 individual
                elif n_inds > 1:
                    for t_ips_1i in list_t_ips:
                        for k,v in t_ips_1i.items():
                            self.assertIsInstance(v, torch.Tensor)
                            self.assertEqual(v.dim(), 2)
                            self.assertEqual(v.shape, (1, src_dim if (k == 'sources') else 1))

    # @TestHelpers.allow_abstract_class_init(AbstractModel)
    # def test_compute_individual_trajectory(self):
    #     # TODO not sure it is the right place to test that
    #     # multivariate
    #     leaspy_object = Leaspy.load(hardcoded_model_path('logistic_scalar_noise'))
    #     abstract_model = AbstractModel("logistic")
    #     abstract_model.load_parameters(leaspy_object.model.parameters)
    #
    #     ip = {
    #       "xi": 0.1,
    #       "tau": 70,
    #       "sources": [
    #         0.1,
    #         -0.3
    #       ]
    #     }
    #
    #     timepoints = [78, 81]
    #
    #     expected_estimation = torch.tensor([
    #         [[0.99641526, 0.34549406, 0.67467, 0.98959327],
    #          [0.9994672, 0.5080943, 0.8276345, 0.99921334]]
    #     ])
    #
    #     indiv_trajectory = abstract_model.compute_individual_trajectory(timepoints, ip)
    #     self.assertEqual(indiv_trajectory.shape, (1, len(timepoints), 2))
    #     self.assertTrue(torch.eq(indiv_trajectory, expected_estimation))
    #
    #     # TODO univariate ?
    #
