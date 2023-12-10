import pytest

@pytest.mark.parametrize("i,j", [(1, 2), (3, 4), (5, 6)])
def test_numbers(i, j):
    assert i != j
