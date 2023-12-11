from jinja2 import nodes
from jinja2.ext import Extension

class MarkdownExtension(Extension):
    tags = set(['markdown'])

    def __init__(self, environment):
        super(MarkdownExtension, self).__init__(environment)

    def parse(self, parser):
        lineno = next(parser.stream).lineno

        # Look for an `extra_attrs` option after the tag name
        args = []
        while not parser.stream.current.test_any('block_end'):
            if parser.stream.skip_if('name:extra_attrs'):
                parser.stream.skip(1)  # skip the equals sign
                value = parser.stream.expect('string').value  # get the string value
                extra_attrs = self.parse_extra_attrs(value)
                args.append(nodes.Const(extra_attrs))
            else:
                parser.stream.next()

        body = parser.parse_statements(['name:endmarkdown'], drop_needle=True)
        return nodes.CallBlock(self.call_method('_render_markdown', args), [], [], body).set_lineno(lineno)

    def parse_extra_attrs(self, value):
        extra_attrs = {}
        for part in value.split():
            tag, attrs = part.split(':')
            extra_attrs[tag] = attrs.split(',')
        return extra_attrs

    def _render_markdown(self, extra_attrs, caller):
        # Use extra_attrs somehow...
        return render_markdown(caller(), extra_attrs)
