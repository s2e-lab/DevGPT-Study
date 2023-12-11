import re
import click
import sys
import tiktoken

@click.command()
@click.version_option()
@click.argument("prompt", nargs=-1)
@click.option("-i", "--input", "input", type=click.File("r"))
@click.option("-t", "--truncate", "truncate", type=int, help="Truncate to this many tokens")
@click.option("-m", "--model", default="gpt-3.5-turbo", help="Which model to use")
@click.option("--tokens", "output_tokens", is_flag=True, help="Output token integers")
@click.option("--decode", "decode", is_flag=True, help="Decode token integers to text")
def cli(prompt, input, truncate, model, output_tokens, decode):
    """
    Count, decode, and truncate text based on tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError as e:
        raise click.ClickException(f"Invalid model: {model}") from e

    if not prompt and input is None:
        input = sys.stdin
    text = " ".join(prompt)
    if input is not None:
        input_text = input.read()
        if text:
            text = input_text + " " + text
        else:
            text = input_text

    if decode:
        # Use regex to find all integers in the input text
        tokens = [int(t) for t in re.findall(r'\d+', text)]
        decoded_text = encoding.decode(tokens)
        click.echo(decoded_text)
    else:
        # Tokenize it
        tokens = encoding.encode(text)
        if truncate:
            tokens = tokens[:truncate]

        if output_tokens:
            click.echo(" ".join(str(t) for t in tokens))
        elif truncate:
            click.echo(encoding.decode(tokens), nl=False)
        else:
            click.echo(len(tokens))
