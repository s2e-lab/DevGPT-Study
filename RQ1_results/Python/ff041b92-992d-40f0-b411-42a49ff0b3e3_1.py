from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader('/path/to/templates/'),
    extensions=[MarkdownExtension]
)
