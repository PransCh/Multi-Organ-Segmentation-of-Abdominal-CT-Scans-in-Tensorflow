from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import os
from utils import main as utils_main

app = Flask(__name__)

# Configuration
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['RESULTS_DIRECTORY'] = 'results/'  # Directory where processed images are stored
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/home')
def home():
    return render_template('temp.html')

@app.route('/upload_page')
def upload_page():
    return render_template('upload_page.html')



@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if the file was uploaded
        if 'imageinput' not in request.files:
            return render_template('temp.html', message='No file selected')

        file = request.files['imageinput']

        # Check if the filename is empty
        if file.filename == '':
            return render_template('upload_page.html', message='No selected file')

        # Check the file extension
        if not allowed_file(file.filename):
            return render_template('temp.html', message='The image is not in the proper format. Please select a .jpg, .jpeg, or .png')

        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_DIRECTORY'], filename)
        file.save(file_path)

        # Check which button was clicked
        upload_button = request.form['uploadbutton']

        # Define variables for UNET or UNETR
        if upload_button == 'unet':
            IMG_H = 320
            IMG_W = 416
            NUM_CLASSES = 11
            model_path = "Unet_100_model.h5" # unet-model
        elif upload_button == 'unetr':
            IMG_H = 256
            IMG_W = 256
            NUM_CLASSES = 11
            model_path = "Ultimate_unter_model160epo.h5" # unetr-model
        else:
            return render_template('temp.html', message='Invalid operation')

        # Call the main function of utils.py with the file path and model path
        CLASSES = utils_main(file_path, model_path, IMG_H, IMG_W, NUM_CLASSES)

        # Return a message or redirect to another page
        return render_template('upload_page.html', message='Image processed successfully!', image_path=file_path)

    except RequestEntityTooLarge:
        return render_template('temp.html', message='File is larger than the 16MB limit')


@app.route('/results/<filename>')
def display_result(filename):
    return send_from_directory(app.config['RESULTS_DIRECTORY'], filename)

if __name__ == '__main__':
    app.run(debug=True)



# from flask import Flask, render_template, request, send_from_directory
# from werkzeug.utils import secure_filename
# import os
# from utils import main

# app = Flask(__name__)

# # Configuration
# app.config['UPLOAD_DIRECTORY'] = 'uploads/'
# app.config['RESULTS_DIRECTORY'] = 'results/'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# # Function to check if file extension is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/')
# def index():
#     return render_template('temp.html')

# @app.route('/upload_page')
# def upload_page():
#     return render_template('upload_page.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     try:
#         # Check if the file was uploaded
#         if 'imageinput' not in request.files:
#             return render_template('temp.html', message='No file selected')

#         file = request.files['imageinput']

#         # Check if the filename is empty
#         if file.filename == '':
#             return render_template('upload_page.html', message='No selected file')

#         # Check the file extension
#         if not allowed_file(file.filename):
#             return render_template('temp.html', message='The image is not in the proper format. Please select a .jpg, .jpeg, or .png')

#         # Save the file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_DIRECTORY'], filename)
#         file.save(file_path)

#         # Determine which model to use based on button click
#         model_path = "Unet_100_model.h5" if request.form.get('model') == 'unet' else "unetr_model_clean.keras"
#         CLASSES = main(file_path, model_path)

#         # Return a message or redirect to another page
#         return render_template('upload_page.html', message='Image processed successfully!', image_path=file_path)

#     except Exception as e:
#         return render_template('upload_page.html', message=f'An error occurred: {str(e)}')

# @app.route('/results/<filename>')
# def display_result(filename):
#     return send_from_directory(app.config['RESULTS_DIRECTORY'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)
