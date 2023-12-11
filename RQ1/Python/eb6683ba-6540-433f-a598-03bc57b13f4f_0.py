from flask import Flask, request, Response
import requests

app = Flask(__name__)

DISCLAIMER = "<p>This page has been modified by a program for demonstration purposes.</p>"

def inject_disclaimer(html_content):
    modified_html = html_content.replace('<body>', f'<body>{DISCLAIMER}', 1)
    return modified_html

def inject_script(html_content):
    script_tag = '<script src="YOUR_JS_FILE_URL"></script>'  # Replace with your JavaScript file URL
    modified_html = html_content.replace('</head>', f'{script_tag}</head>', 1)
    return modified_html

@app.route('/get_webpage')
def get_webpage():
    target_url = request.args.get('url')
    if not target_url:
        return "No target URL provided."

    try:
        response = requests.get(target_url)
        if response.status_code == 200:
            modified_content = inject_disclaimer(response.content.decode('utf-8'))
            modified_content = inject_script(modified_content)
            return Response(modified_content, content_type=response.headers['Content-Type'])
        else:
            return f"Failed to fetch content from {target_url}."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
