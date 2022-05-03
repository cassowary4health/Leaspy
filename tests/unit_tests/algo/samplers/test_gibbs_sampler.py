from itertools import cycle

import torch

from leaspy.io.data.dataset import Dataset
from leaspy.algo.utils.samplers.gibbs_sampler import GibbsSampler

from tests import LeaspyTestCase


class SamplerTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # for tmp handling
        super().setUpClass()

        cls.leaspy = cls.get_hardcoded_model('logistic_scalar_noise')
        cls.data = cls.get_suited_test_data_for_model('logistic_scalar_noise')
        cls.dataset = Dataset(cls.data)

        # GibbsSampler scales so not to change old results
        cls.scale_ind = .1 / GibbsSampler.STD_SCALE_FACTOR_IND
        cls.scale_pop = 5e-3 / GibbsSampler.STD_SCALE_FACTOR_POP

    def test_realization(self):
        realizations = self.leaspy.model.initialize_realizations_for_model(2)
        tau_real = realizations['tau']
        self.assertIsInstance(str(tau_real), str)

    def test_sample(self):
        """
        Test if samples values are the one expected
        """
        # TODO change this instantiation
        n_patients = 17
        n_draw = 50
        temperature_inv = 1.0

        realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # Test with taus (individual parameter)
        var_name = 'tau'
        for sampler in ['Gibbs']:
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_ind, sampler_type=sampler)
            random_draws = []
            for i in range(n_draw):
                gsampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv, attribute_type=None)
                random_draws.append(realizations[var_name].tensor_realizations.clone())

            stack_random_draws = torch.stack(random_draws)
            stack_random_draws_mean = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).mean(dim=0)
            stack_random_draws_std = (stack_random_draws[1:, :, :] - stack_random_draws[:-1, :, :]).std(dim=0)

            self.assertAlmostEqual(stack_random_draws_mean.mean(), 0.0160, delta=0.05)
            self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0861, delta=0.05)

        # Test with g (1D population parameter)
        var_name = 'g'
        for sampler in ['Gibbs', 'FastGibbs', 'Metropolis-Hastings']:
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_pop, sampler_type=sampler)
            # a valid model MCMC toolbox is needed for sampling a population variable (update in-place)
            self.leaspy.model.initialize_MCMC_toolbox()
            random_draws = []
            for i in range(n_draw):
                gsampler.sample(self.dataset, self.leaspy.model, realizations, temperature_inv)  # attribute_type=None would not be used here
                random_draws.append(realizations[var_name].tensor_realizations.clone())

            stack_random_draws = torch.stack(random_draws)
            stack_random_draws_mean = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).mean(dim=0)
            stack_random_draws_std = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).std(dim=0)

            self.assertAlmostEqual(stack_random_draws_mean.mean(), 4.2792e-05, delta=0.05)
            self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0045, delta=0.05)

        # Test with betas (2 dimensional population parameter)
        var_name = 'betas'
        for sampler in ['Gibbs', 'FastGibbs', 'Metropolis-Hastings']:
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_pop, sampler_type=sampler)
            # a valid model MCMC toolbox is needed for sampling a population variable (update in-place)
            self.leaspy.model.initialize_MCMC_toolbox()
            random_draws = []
            for i in range(n_draw):
                gsampler.sample(self.dataset, self.leaspy.model, realizations,
                                temperature_inv)  # attribute_type=None would not be used here
                random_draws.append(realizations[var_name].tensor_realizations.clone())

            stack_random_draws = torch.stack(random_draws)
            stack_random_draws_mean = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).mean(dim=0)
            stack_random_draws_std = (stack_random_draws[1:, :] - stack_random_draws[:-1, :]).std(dim=0)

            self.assertAlmostEqual(stack_random_draws_mean.mean(), 4.2792e-05, delta=0.05)
            self.assertAlmostEqual(stack_random_draws_std.mean(), 0.0045, delta=0.05)

    def test_acceptation(self):
        n_patients = 17
        n_draw = 200

        # Test with tau (0D individual variable)
        var_name = 'tau'
        cst_acceptation = torch.tensor([1.0]*10+[0.0]*7)
        for sampler in ['Gibbs']:
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_ind, sampler_type=sampler)

            for i in range(n_draw):
                gsampler._update_acceptation_rate(cst_acceptation)

            acc_mean = gsampler.acceptation_history.mean(dim=0)
            self.assertEqual(acc_mean.shape, cst_acceptation.shape)
            self.assertAllClose(acc_mean, cst_acceptation)

        # Test with sources (1D individual variable of dim Ns, here 2) --> we do not take care of dimension for individual parameter!
        var_name = 'sources'
        cst_acceptation = torch.tensor([1.0]*7+[0.0]*10)
        for sampler in ['Gibbs']:
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_ind, sampler_type=sampler)

            for i in range(n_draw):
                gsampler._update_acceptation_rate(cst_acceptation)

            acc_mean = gsampler.acceptation_history.mean(dim=0)
            self.assertEqual(acc_mean.shape, cst_acceptation.shape)
            self.assertAllClose(acc_mean, cst_acceptation)

        # Test with g (1D population variable of dim N, here 4)
        var_name = 'g'
        acceptation_for_draws = {
            'Gibbs': (cycle([torch.tensor([0., 0., 1., 1.])]*3 + [torch.tensor([0., 1., 0., 1.])]*2), torch.tensor([0., 2/5, 3/5, 1.])),
            'FastGibbs': (cycle([torch.tensor([0., 0., 1., 1.])]*3 + [torch.tensor([0., 1., 0., 1.])]*2), torch.tensor([0., 2/5, 3/5, 1.])),
            'Metropolis-Hastings': (cycle([torch.tensor(1.)]*3 + [torch.tensor(0.)]*2), torch.tensor(3/5)),
        }
        for sampler, (acceptation_it, expected_mean_acceptation) in acceptation_for_draws.items():
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_pop, sampler_type=sampler)

            for i in range(n_draw):
                gsampler._update_acceptation_rate(next(acceptation_it))

            acc_mean = gsampler.acceptation_history.mean(dim=0)
            self.assertEqual(acc_mean.shape, expected_mean_acceptation.shape)
            self.assertAllClose(acc_mean, expected_mean_acceptation, msg=(var_name, sampler))

        # Test with betas (2D population variable of dim (N-1, Ns), here (3, 2))
        var_name = 'betas'
        acceptation_for_draws = {
            'Gibbs': (cycle([torch.tensor([[0., 0.], [0., 1.], [1., 1.]])]*3 + [torch.tensor([[0., 1.], [1., 0.], [0., 1.]])]*2), torch.tensor([[0., 2/5], [2/5, 3/5], [3/5, 1.]])),
            'FastGibbs': (cycle([torch.tensor([0., 0., 1.])]*3 + [torch.tensor([0., 1., 0.])]*2), torch.tensor([0., 2/5, 3/5])),
            'Metropolis-Hastings': (cycle([torch.tensor(1.)]*3 + [torch.tensor(0.)]*2), torch.tensor(3/5)),
        }
        for sampler, (acceptation_it, expected_mean_acceptation) in acceptation_for_draws.items():
            gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients,
                                    scale=self.scale_pop, sampler_type=sampler)

            for i in range(n_draw):
                gsampler._update_acceptation_rate(next(acceptation_it))

            acc_mean = gsampler.acceptation_history.mean(dim=0)
            self.assertEqual(acc_mean.shape, expected_mean_acceptation.shape)
            self.assertAllClose(acc_mean, expected_mean_acceptation, msg=(var_name, sampler))

    def test_adaptative_proposition_variance(self):
        n_patients = 17
        n_draw = 200
        # temperature_inv = 1.0

        # realizations = self.leaspy.model.initialize_realizations_for_model(n_patients)

        # Test with taus
        var_name = 'tau'
        gsampler = GibbsSampler(self.leaspy.model.random_variable_informations()[var_name], n_patients, scale=self.scale_ind)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([1.0]*10+[0.0]*7, dtype=torch.float32))

        for i in range(1000):
            gsampler._update_std()

        self.assertAlmostEqual(gsampler.std[:10].mean(), 4.52, delta=0.05)
        self.assertAlmostEqual(gsampler.std[10:].mean(), 0.0015, delta=0.05)

        for i in range(n_draw):
            gsampler._update_acceptation_rate(torch.tensor([0.0]*10+[1.0]*7, dtype=torch.float32))

        for i in range(2000):
            gsampler._update_std()

        self.assertAlmostEqual(gsampler.std[:10].mean(), 9.8880e-04, delta=0.05)
        self.assertAlmostEqual(gsampler.std[10:].mean(), 3.0277, delta=0.05)
