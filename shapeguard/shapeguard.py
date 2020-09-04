import torch as t


class ShapeGuard:
    shapes = dict()

    @staticmethod
    def reset(dims=None):
        if dims is None:
            ShapeGuard.shapes = dict()
        else:
            for d in dims:
                if d in ShapeGuard.shapes:
                    del ShapeGuard.shapes[d]

    @staticmethod
    def assert_shape(actual_shape: t.Size, expected_shape):
        assert len(actual_shape) == len(expected_shape), f"expected {len(expected_shape)} dimensions but tensor had {len(actual_shape)}"
        for d, e in zip(actual_shape, expected_shape):
            if isinstance(e, int):
                assert e == d, f"expected {e} but was {d}"
            elif e == '*':
                continue
            elif e in ShapeGuard.shapes:
                assert ShapeGuard.shapes[e] == d, f"expected '{e}' to be {ShapeGuard.shapes[e]} but was {d}"
            else:
                ShapeGuard.shapes[e] = d

    @staticmethod
    def assert_tensor_shape(tensor: t.Tensor, expected_shape) -> t.Tensor:
        ShapeGuard.assert_shape(tensor.shape, expected_shape)
        return tensor

    @staticmethod
    def assert_distribution_shape(dist: t.distributions.Distribution, expected_shape) -> t.distributions.Distribution:
        ShapeGuard.assert_shape(dist.batch_shape, expected_shape)
        return dist


t.Tensor.sg = ShapeGuard.assert_tensor_shape
t.distributions.Distribution.sg = ShapeGuard.assert_distribution_shape
