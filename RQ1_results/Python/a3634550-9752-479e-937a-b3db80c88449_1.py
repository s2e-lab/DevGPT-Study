from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the request contains a file
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']

        # If the user does not select a file, the browser may submit an empty part without a filename
        if file.filename == '':
            return 'No selected file'

        # Save the uploaded file to a specific location (you can customize this)
        file.save('/path/to/your/upload/folder/' + file.filename)

        # Or, you can process the uploaded file as needed
        # For example, read the image using OpenCV
        # img = cv2.imread('/path/to/your/upload/folder/' + file.filename)

        return 'File uploaded successfully'

if __name__ == '__main__':
    app.run(debug=True)
