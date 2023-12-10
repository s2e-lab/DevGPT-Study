from flask import Flask, render_template_string
import os
from os.path import join
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    path = "/path/to/your/directory"
    files = [(f, datetime.fromtimestamp(os.path.getmtime(join(path, f))))
             for f in os.listdir(path)]

    # Sort by last modified time
    files.sort(key=lambda x: x[1], reverse=True)

    html = "<ul>\n"
    for f, _ in files:
        html += f"<li><a href='{f}'>{f}</a></li>\n"
    html += "</ul>"

    return render_template_string(html)

if __name__ == '__main__':
    app.run(port=8000)
