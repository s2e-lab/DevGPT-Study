from lark import Lark, Tree, Token

grammar = """
    start: word " " word " " word
    word: "cat" | "dog" | "fish"
    %import common.WS
    %ignore WS
"""

def get_next_valid_symbols(parser, input_str):
    try:
        # Try to parse the input string
        parser.parse(input_str)
        # If the input string is valid, then there are no next valid symbols
        return []
    except Exception as e:
        # Get the expected tokens from the exception message
        expected_tokens = str(e).split(':')[1].strip().split(', ')
        return expected_tokens

def main():
    parser = Lark(grammar, start='start')
    input_str = "cat dog"
    next_valid_symbols = get_next_valid_symbols(parser, input_str)
    print("Next valid symbols:", next_valid_symbols)

if __name__ == "__main__":
    main()
