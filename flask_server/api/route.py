from flask import render_template, request, redirect, Blueprint
from flask import jsonify
import torch
from IPython.display import Image, clear_output
import datetime
import os
import traceback
from werkzeug.utils import secure_filename
from flask_server.detect import detect as detect_ai
from flask_server import app

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'mp4'])
todo = Blueprint('todo', __name__, url_prefix='/')

@todo.route('/', methods=['POST','GET'])
def index():
    return render_template('upload.html')


@todo.route('/checkGPU', methods=['GET'])
def checkGPU():
    torch_ver ='%s' % (torch.__version__)
    gpu ='%s' % (torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU')
    return jsonify({"torch_ver": torch_ver, "gpu":gpu})


def detect_barcode(src_path):
    with torch.no_grad():
        save_path = detect_ai(src_path, img_size=640)
    return save_path


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@todo.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        try:
            real_path = os.path.realpath('') 
            
            # check if the post request has the file part
            if 'file' not in request.files:
                return "File Empty", 401

            file = request.files['file']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return "File name Empty", 401

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = str(filename).split('.')
                time_stamp = int(datetime.datetime.utcnow().timestamp())
                filename = filename[0] + '_' + str(time_stamp) + '.' +filename[1].lower()
                save_path = os.path.join(real_path, app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_path = detect_barcode(save_path).replace('\\','/')
                image_path = image_path.replace('flask_server/static/','')

            return render_template('upload.html', image = image_path)
        except Exception as e:
            traceback.print_exc()
            return "Server Error", 403
