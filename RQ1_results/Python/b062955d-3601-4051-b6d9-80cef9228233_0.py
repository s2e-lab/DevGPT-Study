import pytest
from my_module import read_file  # Replace with the actual module name

# Define the test cases
test_cases = [
    ("/mnt/data/utf8_file.txt", "# coding: utf-8\nThis is a text file."),
    ("/mnt/data/latin1_file.txt", "# coding: latin1\nThis is a text file."),
    ("/mnt/data/ascii_file.txt", "# coding: ascii\nThis is a text file."),
    ("/mnt/data/no_decl_file.txt", "This is a text file."),
    ("/mnt/data/invalid_decl_file.txt", "# coding: invalid\nThis is a text file."),
    ("/mnt/data/empty_file.txt", ""),
]

@pytest.mark.parametrize("filepath, expected", test_cases)
def test_read_file(filepath, expected):
    assert read_file(filepath) == expected
