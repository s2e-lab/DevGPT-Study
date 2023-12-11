from parameterized import parameterized
import unittest

class MyTest(unittest.TestCase):
    @parameterized.expand([
        (1, 2),
        (3, 4),
        (5, 6),
    ])
    def test_numbers(self, i, j):
        self.assertNotEqual(i, j)
