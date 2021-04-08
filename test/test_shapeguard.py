import unittest
import torch as t
from shapeguard import ShapeGuard


class MyTestCase(unittest.TestCase):
    def test_returns_tensor_if_shapes_matches(self):
        ShapeGuard.reset()
        a = t.zeros((3, 2))
        self.assertTrue(a is a.sg("ab"))

    def test_returns_tensor_if_shapes_matches_exactly(self):
        ShapeGuard.reset()
        a = t.zeros((3, 2))
        self.assertTrue(a is a.sg(("a", 2)))

    def test_returns_tensor_if_shapes_matches_ignores_asterix(self):
        ShapeGuard.reset()
        a = t.zeros((3, 2))
        self.assertTrue(a is a.sg("a*"))

    def test_asserts_tensor_shapes_matches(self):
        ShapeGuard.reset()
        a = t.zeros((3, 2))
        b = t.zeros((3, 2, 4))
        self.assertTrue(a is a.sg("ab"))
        self.assertTrue(b is b.sg("abc"))

    def test_raises_if_wrong_num_dimensions(self):
        ShapeGuard.reset()
        self.assertRaises(AssertionError, lambda: t.zeros((3, 2)).sg("z"))

    def test_raises_if_two_tensors_dont_match(self):
        ShapeGuard.reset()
        a = t.zeros((3, 2))
        b = t.zeros((3, 4))
        self.assertTrue(a is a.sg("ab"))
        self.assertRaises(AssertionError, lambda: b.sg("ab"))


if __name__ == '__main__':
    unittest.main()
