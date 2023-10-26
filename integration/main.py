from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'
ANOMALY_FOLDER_HARD_PATH = 'D:/edgehack/juicebox-behind/integration/static/all_anomalies'
UPLOAD_FOLDER = './static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

ANOMALY_FOLDER = './static/all_anomalies'
app.config['ANOMALY_FOLDER'] = ANOMALY_FOLDER
if not os.path.exists(app.config['ANOMALY_FOLDER']):
    os.makedirs(app.config['ANOMALY_FOLDER'])
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


import cv2
from utils import generate_sections, find_anomalies

def get_sections(reference_path,multi_up_path):
    # T1. Generate Sections
    # ref = '67'
    # reference_path = f'./A0{ref}/SA-A0{ref}.jpg'
    # multi_up_path = "multi_up.bmp"
    expected_dim = (830, 1073) # (widht, height) of the single expected image

    reference_image = cv2.imread(reference_path, cv2.IMREAD_COLOR)
    multi_up_image = cv2.imread(multi_up_path, cv2.IMREAD_COLOR)
    reference_image = cv2.resize(reference_image, expected_dim)

    # None is the output_path if save it is true it has to be defined
    # rotation matters because I am extruding the sections from the multi_up image
    _,sections,bb_img,r,_,_ = generate_sections(reference_image, multi_up_image, None,threshold=0.8)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'multiup_bbs.jpg'), bb_img)
    # _,sections,bb_img,_,_,_ = generate_sections(reference_image, multi_up_image, None)
    print(r.max())
    return bb_img, sections


source_image_paths = []
template_image_paths = []
sections = []
@app.route('/process_images', methods=['POST'])
def process_images():

    template_image = request.files['template_image']
    source_image = request.files['source_image']
    template_image_path = os.path.join(app.config['UPLOAD_FOLDER'], template_image.filename)
    source_image_path = os.path.join(app.config['UPLOAD_FOLDER'], source_image.filename)
    template_image.save(template_image_path)
    source_image.save(source_image_path)
    print("Images saved")
    source_image_paths.append(source_image_path)
    template_image_paths.append(template_image_path)

    # crop out white padding
    image = cv2.imread(template_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    image = image[y:y+h, x:x+w]
    cv2.imwrite(template_image_path, image)

    bb_img,sections_returned = get_sections(template_image_path,source_image_path)
    sections.append(sections_returned)

    return render_template('results.html', template_image_path=template_image_path, bb_source_image_path="static/uploads/multiup_bbs.jpg")

@app.route('/process_images', methods=['GET'])
def process_images_get():
    return redirect(url_for('results'))

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_anomalies(sections):
    # T2. Anomaly detection
    anomalies_in_all_matched_sections=[]
    for section in sections:
        anomalies_in_a_section=find_anomalies(section)
        anomalies_in_all_matched_sections.append(anomalies_in_a_section)
    return anomalies_in_all_matched_sections

def save_anomalies_bbs(all_anomalies,source_image_path,template_image_path):
    # T3. Anomaly localization
    expected_dim = (830, 1073) # (widht, height) of the single expected image
    reference_image = cv2.imread(template_image_path, cv2.IMREAD_COLOR)
    multi_up_image = cv2.imread(source_image_path, cv2.IMREAD_COLOR)
    reference_image = cv2.resize(reference_image, expected_dim)    
    for i,anomalies in enumerate(all_anomalies):
        for j,anomaly in enumerate(anomalies):
            x,y=anomaly
            w,h=166,166

            # crop this from the multi_up_image and save it
            tl=(x,y)
            br=(x+w,y+h)
            anomaly_img = multi_up_image[tl[1]:br[1],tl[0]:br[0]]
            cv2.imwrite(f"{ANOMALY_FOLDER_HARD_PATH}/{i}-{j}.jpg",anomaly_img)

@app.route('/comparison', methods=['GET'])
def comparison():
    all_anomalies = get_anomalies(sections[0])
    save_anomalies_bbs(all_anomalies,source_image_paths[0],template_image_paths[0])
    # anomaly_files = [f for f in os.listdir(ANOMALY_FOLDER_HARD_PATH) if os.path.isfile(os.path.join(app.config['ANOMALY_FOLDER'], f))]
    anomaly_files = [f for f in os.listdir(ANOMALY_FOLDER_HARD_PATH)]

    return render_template("comparison.html", anomalies=anomaly_files)

@app.route('/test', methods=['GET'])
def test():
    # all_anomalies = get_anomalies(sections[0])
    # save_anomalies_bbs(all_anomalies,source_image_paths[0],template_image_paths[0])
    # anomaly_files = [f for f in os.listdir(ANOMALY_FOLDER_HARD_PATH) if os.path.isfile(os.path.join(app.config['ANOMALY_FOLDER'], f))]
    anomaly_files = [f for f in os.listdir(ANOMALY_FOLDER_HARD_PATH)]

    return render_template("comparison.html", anomalies=anomaly_files)



if __name__ == '__main__':
    app.run(debug=True)





# # T3. Anomaly localization
