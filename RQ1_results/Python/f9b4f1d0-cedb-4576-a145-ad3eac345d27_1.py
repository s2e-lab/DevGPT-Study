import pytest
from mymodule import convert_to_quoted  # Assuming the function is in `mymodule.py`

def test_convert_to_quoted():
    # Single line, no quotes
    assert convert_to_quoted("Hello, World!") == '"Hello, World!"'
    
    # Single line, with quotes
    assert convert_to_quoted('Hello, "World"!') == '"Hello, \\"World\\"!"'
    
    # Multiline, no quotes
    multiline_str = "Hello,\nWorld!"
    expected_result = '"""Hello,\nWorld!"""'
    assert convert_to_quoted(multiline_str) == expected_result
    
    # Multiline, with triple quotes
    multiline_str = '''Hello,
"World",
Here are some triple quotes: """ '''
    expected_result = '"""Hello,\n\\"World\\",\nHere are some triple quotes: \\"\\"\\" """'
    assert convert_to_quoted(multiline_str) == expected_result

    # Empty string
    assert convert_to_quoted('') == '""'
