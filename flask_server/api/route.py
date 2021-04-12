import torch
import datetime
import os
import traceback
import threading 
import glob
import random
import shutil
import cv2
from datetime import datetime

from IPython.display import Image, clear_output

from flask_cors import cross_origin
from flask import render_template, request, redirect, Blueprint, Response
from flask import jsonify
from werkzeug.utils import secure_filename

from flask_server.detect import detect as detect_ai
from flask_server import app

from utils.render_video import set_fps, resize_video
from utils.read_code import read_barcode
from utils.azure_service import uploadVideo as azure_upload




ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])
detect = Blueprint('detect', __name__, url_prefix='/')

@detect.route('/', methods=['POST','GET'])
def index():
    return render_template('index.html')


def detect_barcode(src_path):
    with torch.no_grad():
        save_path = detect_ai(src_path, img_size=640)
    return save_path


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file):
    save_path = ''
    file_type = 'images'
    real_path = os.path.realpath('') 
    try:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename_sl = str(filename).split('.')
            time_stamp = int(datetime.datetime.utcnow().timestamp())
            filename = filename_sl[0] + '_' + str(time_stamp) + '.' +filename_sl[1].lower()
            if filename_sl[1].lower() == 'mp4':
                file_type = 'videos'
            save_path = os.path.join(real_path, app.config['UPLOAD_FOLDER'], file_type, filename)
            print(save_path)
            file.save(save_path)
    except:
         traceback.print_exc()
    finally:
        return save_path, file_type


@detect.route('/upload', methods=['POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        try:

            # check if the post request has the file part
            if 'files' not in request.files:
                return "File Empty", 401

            files = request.files.getlist("files")
            list_result = []

            for file in files:
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    return 'File name is empty', 401
                else:
                    save_path, file_type = save_file(file)

                    if save_path != '':
                        if file_type == 'videos':
                            save_path = set_fps(save_path, 7, '_edit')
                        file_path, list_barcode = detect_barcode(save_path)
                        if file_type == 'videos':
                            file_path = resize_video(file_path, (1280, 720),'_papv')
                        # token = app.config['FIREBASE_STORAGE'].child(file.filename).put(file_path)["downloadTokens"]
                        # link = app.config['FIREBASE_STORAGE'].child(file.filename).get_url(token)
                        
                        # current date and time
                        now = datetime.now()
                        timestamp = datetime.timestamp(now)
                        name_upload = str(file.filename).replace('.', '_'str(timestamp) + '.')
                        link = azure_upload('videos', file_path, app.config['AZURE_CONN_STR'], name_upload)
                        # copy_file_label(file_path)
                        list_result.append({'link':link, 'list_code':list_barcode})
                    else:
                        return 'Server Error', 500
            return jsonify(list_result), 200
        except Exception as e:
            traceback.print_exc()
            return "Server Error", 403


def copy_file_label(file_path):
    path_split = file_path.split('\\')
    filename = path_split[-1]
    file_type = check_file_type(filename)
    filename = filename.replace('.'+file_type, '.txt')

    path_split[-1] = 'labels'
    label_dir = '\\'.join(path_split)
    
    if file_type != 'mp4':
        path = os.path.join(label_dir, filename)
        path_label = os.path.join(app.config['FOLDER_LABEL'], filename)
        shutil.copyfile(path, path_label)
    else:
        labels =  glob.glob(label_dir+'\\*.txt')
        for i in range(len(labels)):
            r = i + 1
            path = labels[i]
            name = filename.replace('.', '_%s.'%r)
            path_label = os.path.join(app.config['FOLDER_LABEL'], name)
            shutil.copyfile(path, path_label)


def check_file_type(filename):
    return filename.split('.')[-1].lower()

def distribute_data():
    real_path = os.path.realpath('') 
    path_label = os.path.join(real_path, app.config['FOLDER_LABEL'])
    labels =  glob.glob(path_label+'\\*.txt')

    storage_image = os.path.join(real_path, app.config['UPLOAD_FOLDER'])

    to_val_lab = '\\retrain\\val\\labels\\'
    to_train_lab = '\\retrain\\train\\labels\\'
    to_val_img = '\\retrain\\val\\images\\'
    to_train_img = '\\retrain\\train\\images\\'

    random.shuffle(labels)
    train = int(len(labels) * 0.8)
    val = len(labels) - train

    for path in labels[:train]:
        to_train_lab = path.replace('\\labels\\', to_train_lab)
        shutil.copyfile(path, to_train_lab)

        img_path = storage_image + '\\' + get_filename_from_path(path).replace('.txt','.jpg')
        to_train_img = img_path.replace('\\upload\\images\\', to_train_img)
        shutil.copyfile(img_path, to_train_img)

    for path in labels[train:]:
        to_train_lab = path.replace('\\labels\\', to_train_lab)
        shutil.copyfile(path, to_train_lab)

        img_path = storage_image + '\\' + get_filename_from_path(path).replace('.txt','.jpg')
        to_train_img = img_path.replace('\\upload\\images\\', to_train_img)
        shutil.copyfile(img_path, to_train_img)


def get_filename_from_path(file_path):
    return file_path.rsplit('\\')


@detect.route('/stream')
@cross_origin()
def stream():
    app.config["STREAM"] = True
    t1 = threading.Thread(target=detect_barcode, args=('0',))
    t1.start()
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    while app.config["STREAM"]:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('stream/demo.jpg', 'rb').read() + b'\r\n')
        sleep(1.2)


@detect.route('/stop-stream', methods=['GET'])
@cross_origin()
def stop_stream():
    app.config['CAP_VIDEO'] = None
    app.config['STREAM'] = False
    return "OK", 200

