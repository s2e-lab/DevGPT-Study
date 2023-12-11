import panel as pn
from flask import Flask, render_template_string
pn.extension()

# This function simulates your bot's response
def get_bot_response(user_input):
    return f"You said: {user_input}"

# Create Panel widgets
text_input = pn.widgets.TextInput(name="Input", placeholder="Type here...")
output = pn.pane.Markdown('Bot says: Hi')

@pn.depends(text_input.param.value, watch=True)
def update_output(event):
    bot_response = get_bot_response(event.new)
    output.object = f'Bot says: {bot_response}'

# Create a Panel layout
layout = pn.Column(text_input, output)

# Define Flask server
app = Flask(__name__)

@app.route('/')
def panel_app():
    # Convert Panel app to HTML and serve via Flask
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    </head>
    <body>
        {{ embed(roots.layout) }}
    </body>
    </html>
    """, embed=pn.pane.HTML(layout._get_root().get_root().embed(max_opts=1)))


if __name__.startswith("bk"):
    layout.servable()
elif __name__ == "__main__":
    print('Opening single process Flask app with embedded Panel application on http://localhost:5000/')
    print('Multiple connections may block the Bokeh server and cause unresponsive behavior.')
    app.run(port=5000)
