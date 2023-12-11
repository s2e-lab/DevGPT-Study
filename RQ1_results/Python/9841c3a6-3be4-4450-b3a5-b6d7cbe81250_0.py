import unittest

class MyTest(unittest.TestCase):
    def test_numbers(self):
        for i, j in [(1, 2), (3, 4), (5, 6)]:
            with self.subTest(i=i, j=j):
                self.assertNotEqual(i, j)
