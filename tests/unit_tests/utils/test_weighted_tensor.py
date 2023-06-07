import torch
import numpy as np


def test_weighted_tensor_instantiation_without_weights():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0])

    assert torch.equal(t.value, torch.Tensor([-1.0, 0.0, 1.0]))
    assert t.weight is None
    assert t.shape == torch.Size([3])
    assert t.ndim == 1
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert not t.requires_grad


def test_weighted_tensor_instantiation_with_weights():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0], [0, 1, 1])

    assert torch.equal(t.value, torch.Tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(t.weight, torch.tensor([0, 1, 1], dtype=torch.int64))
    assert t.shape == torch.Size([3])
    assert t.ndim == 1
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert not t.requires_grad


def test_weighted_tensor_instantiation_from_numpy_array_without_weights():
    from leaspy.utils.weighted_tensor import WeightedTensor

    data = np.random.random((3, 6, 9))
    t = WeightedTensor(data)

    assert torch.equal(t.value, torch.from_numpy(data))
    assert t.weight is None
    assert t.shape == torch.Size([3, 6, 9])
    assert t.ndim == 3
    assert t.dtype == torch.float64
    assert t.device.type == "cpu"
    assert not t.requires_grad


def test_weighted_tensor_instantiation_from_numpy_array_with_weights():
    from leaspy.utils.weighted_tensor import WeightedTensor

    data = np.random.random((3, 6, 9))
    weights = np.random.random((3, 6, 9))
    t = WeightedTensor(data, weights)

    assert torch.equal(t.value, torch.from_numpy(data))
    assert torch.equal(t.weight, torch.from_numpy(weights))
    assert t.shape == torch.Size([3, 6, 9])
    assert t.ndim == 3
    assert t.dtype == torch.float64
    assert t.device.type == "cpu"
    assert not t.requires_grad


def test_weighted_tensor_operation():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0])
    expected = WeightedTensor([0.0, 1.0, 2.0])

    assert t + 1 == expected
    assert t - (-torch.ones(())) == expected
    assert 1 + (-torch.ones(t.shape)) * (-t) == expected
    assert torch.ones(t.shape) - t * (-1) == expected


def test_weighted_tensor_operation_between_weighted_tensors():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0])
    t1 = WeightedTensor(t.value, abs(t).value.to(int))
    t2 = WeightedTensor(t.value, t1.weight)

    assert t1 - t2 == 0
    assert t1 + t2 == 2 * t1
    assert -t1 + 2 * t2 == t2
    assert t1 * t2 == t1**2


def test_weighted_tensor_identities():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0])
    t1 = WeightedTensor(t.value, abs(t).value.to(int))
    t2 = WeightedTensor(t.value, t1.weight)
    old_id1 = id(t1)

    assert t1 is not t2
    assert t1.value is t2.value

    t1 -= t2
    assert id(t1) != old_id1
    assert t1.value is not t2.value


def test_weighted_tensor_comparisons():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([-1.0, 0.0, 1.0])
    t1 = WeightedTensor(t.value, abs(t).value.to(int))
    t2 = WeightedTensor(t.value, t1.weight)

    assert t1 == t2
    assert t1 - 0.2 <= t2
    assert t1 - 0.2 < t2
    assert t1 + t2 > t1
    t1 -= t2
    assert t2 >= t1


def test_weighted_tensor_filled():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([[-1.0, 0.0, 1.0], [3.14, 42.1, 16.6]])
    t1 = WeightedTensor(t.value, [[1, 0, 0], [1, 0, 1]])

    assert torch.equal(t1.filled(None), t1.value)
    assert torch.equal(t.filled(0.0), t.value)
    assert torch.equal(t1.filled(0.0), torch.tensor([[-1.0, 0.0, 0.0], [3.14, 0.0, 16.6]]))


def test_weighted_tensor_valued():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([[-1.0, 0.0, 1.0], [3.14, 42.1, 16.6]])
    t1 = WeightedTensor(t.value, [[1, 0, 0], [1, 0, 1]])
    t2 = t1.valued(torch.ones_like(t.value))

    assert torch.equal(t2.value, torch.tensor([[1., 1., 1.], [1., 1., 1.]]))
    assert torch.equal(t2.weight, torch.tensor([[1, 0, 0], [1, 0, 1]]))


def test_weighted_tensor_map():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([[-1.0, 0.0, 1.0], [3.14, 42.1, 16.6]])
    t1 = WeightedTensor(t.value, [[1, 0, 0], [1, 0, 1]])

    def func(x, y):
        return x * 6 + y

    t2 = t1.map(func, torch.ones((2, 3)))

    assert torch.allclose(t2.value, torch.tensor([[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]]))
    assert torch.equal(t2.weight, torch.tensor([[1, 0, 0], [1, 0, 1]]))

    t3 = t1.map(func, torch.ones((2, 3)), fill_value=100.0)

    assert torch.allclose(t3.value, torch.tensor([[-5.0, 601.0, 601.0], [19.84, 601.0, 100.6]]))
    assert torch.equal(t3.weight, torch.tensor([[1, 0, 0], [1, 0, 1]]))

    t4 = t.map(func, torch.ones((2, 3)), fill_value=200)

    assert torch.allclose(t4.value, torch.tensor([[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]]))
    assert t4.weight is None


def test_weighted_tensor_map_both():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor([[-1.0, 0.0, 1.0], [3.14, 42.1, 16.6]])
    t1 = WeightedTensor(t.value, [[1, 0, 0], [1, 0, 1]])

    def func(x, y):
        return x * 6 + y

    t5 = t1.map_both(func, torch.ones((2, 3)))

    assert torch.allclose(t5.value, torch.tensor([[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]]))
    assert torch.equal(t5.weight, torch.tensor([[7., 1., 1.], [7., 1., 7.]]))

    t6 = t1.map_both(func, torch.ones((2, 3)), fill_value=100.0)

    assert torch.allclose(t6.value, torch.tensor([[-5.0, 601.0, 601.0], [19.84, 601.0, 100.6]]))
    assert torch.equal(t5.weight, torch.tensor([[7., 1., 1.], [7., 1., 7.]]))

    t7 = t.map_both(func, torch.ones((2, 3)), fill_value=200)

    assert torch.allclose(t7.value, torch.tensor([[-5.0, 1.0, 7.0], [19.84, 253.6, 100.6]]))
    assert t7.weight is None


def test_weighted_tensor_wsum():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor(
        [
            [-1.0, 0.0, 1.0],
            [3.14, 42.1, 16.6],
            [102.2, -23.1, -0.2],
        ],
        [
            [1, 0, 0],
            [1, 0, 2],
            [0, 0, 0],
        ],
    )

    x, y = t.wsum()
    z = t.sum()

    assert torch.equal(x, torch.tensor(35.3400))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor(4))

    x, y = t.wsum(axis=0)
    z = t.sum(axis=0)

    assert torch.equal(x, torch.tensor([2.14, 0.0, 33.2]))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor([2, 0, 2]))

    x, y = t.wsum(axis=1)
    z = t.sum(axis=1)

    assert torch.equal(x, torch.tensor([-1.0, 36.34, 0.0]))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor([1, 3, 0]))

    x, y = t.wsum(fill_value=1.5)
    z = t.sum(fill_value=1.5)

    assert torch.equal(x, torch.tensor(35.3400))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor(4))

    x, y = t.wsum(fill_value=1.5, axis=0)
    z = t.sum(fill_value=1.5, axis=0)

    assert torch.equal(x, torch.tensor([2.14, 1.5, 33.2]))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor([2, 0, 2]))

    x, y = t.wsum(fill_value=1.5, axis=1)
    z = t.sum(fill_value=1.5, axis=1)

    assert torch.equal(x, torch.tensor([-1.0, 36.34, 1.5]))
    assert torch.equal(x, z)
    assert torch.equal(y, torch.tensor([1, 3, 0]))


def test_weighted_tensor_view():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor(
        [
            [-1.0, 0.0, 1.0],
            [3.14, 42.1, 16.6],
            [102.2, -23.1, -0.2],
        ],
        [
            [1, 0, 0],
            [1, 0, 2],
            [0, 0, 0],
        ],
    )

    s = t.view((1, 9))

    assert torch.equal(
        s.value,
        torch.tensor([[-1.0000, 0.0000, 1.0000, 3.1400, 42.1000, 16.6000, 102.2000, -23.1000, -0.2000]])
    )
    assert torch.equal(s.weight, torch.tensor([[1, 0, 0, 1, 0, 2, 0, 0, 0]]))

    s = t.view((9, 1))

    assert torch.equal(
        s.value,
        torch.tensor(
            [
                [-1.0000],
                [0.0000],
                [1.0000],
                [3.1400],
                [42.1000],
                [16.6000],
                [102.2000],
                [-23.1000],
                [-0.2000],
            ]
        )
    )
    assert torch.equal(s.weight, torch.tensor([[1], [0], [0], [1], [0], [2], [0], [0], [0]]))


def test_weighted_tensor_expand():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = WeightedTensor(
        [[-1.0, 0.0, 1.0]],
        [[1, 0, 0]],
    )

    s = t.expand((3, -1))

    assert torch.equal(
        s.value,
        torch.tensor(
            [
                [-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.],
            ]
        )
    )
    assert torch.equal(
        s.weight,
        torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ]
        )
    )


def test_get_filled_value_and_weight():
    from leaspy.utils.weighted_tensor import WeightedTensor

    t = torch.tensor(
        [
            [-1.0, 0.0, 1.0],
            [3.14, 42.1, 16.6],
            [102.2, -23.1, -0.2],
        ]
    )
    x, y = WeightedTensor.get_filled_value_and_weight(t)

    assert torch.equal(x, t)
    assert y is None

    x, y = WeightedTensor.get_filled_value_and_weight(t, fill_value=1000.0)

    assert torch.equal(x, t)
    assert y is None

    t2 = WeightedTensor(
        [
            [-1.0, 0.0, 1.0],
            [3.14, 42.1, 16.6],
            [102.2, -23.1, -0.2],
        ],
        [
            [1, 0, 0],
            [1, 0, 2],
            [0, 0, 0],
        ],
    )

    x, y = WeightedTensor.get_filled_value_and_weight(t2)

    assert torch.equal(x, t)
    assert torch.equal(y, torch.tensor([[1, 0, 0], [1, 0, 2], [0, 0, 0]]))

    x, y = WeightedTensor.get_filled_value_and_weight(t2, fill_value=1000.0)

    assert torch.equal(
        x,
        torch.tensor(
            [
                [-1.0, 1000.0, 1000.0],
                [3.14, 1000.0, 16.6],
                [1000.0, 1000.0, 1000.0],
            ]
        )
    )
    assert torch.equal(y, torch.tensor([[1, 0, 0], [1, 0, 2], [0, 0, 0]]))
