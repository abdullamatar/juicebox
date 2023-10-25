from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images():
    if 'template_image' not in request.files or 'source_image' not in request.files:
        flash('Please upload both template and source images.')
        return redirect(url_for('index'))

    template_image = request.files['template_image']
    source_image = request.files['source_image']

    if template_image.filename == '' or source_image.filename == '':
        flash('Please select valid files for both images.')
        return redirect(url_for('index'))

    if allowed_file(template_image.filename) and allowed_file(source_image.filename):
        # Save the uploaded images to the 'uploads' directory
        template_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'template_image.jpg')
        source_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'source_image.jpg')
        template_image.save(template_image_path)
        source_image.save(source_image_path)

        # Implement image processing and anomaly detection here

        return render_template('results.html')
    else:
        flash('Invalid file format. Please use jpg, jpeg, png, or gif.')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
