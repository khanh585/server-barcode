
from utils.torch_utils import select_device
from utils.upload_file import init_storage
from models.experimental import attempt_load
from flask_cors import CORS, cross_origin
import torch
import os

# Initialize
device = select_device('')
# Load model

real_path = os.path.realpath('')
weight_path = os.path.join(real_path, r'flask_server\\weights\last.pt')
model = attempt_load(weight_path, map_location=device) 

from flask import Flask


app = Flask(__name__)
cors = CORS(app)

app.config['UPLOAD_FOLDER'] = r'flask_server\storage\uploads'
app.config['FOLDER_LABEL'] = r'flask_server\storage\labels'
app.config['WEIGHT_FILES'] = r'flask_server\weights\last.pt'
app.config['SAVE_IMAGES_PATH'] = r'flask_server\static\detect'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MODEL'] = model
app.config['DEVICE'] = device
app.config['FIREBASE_STORAGE'] = init_storage()
app.config['STREAM'] = False
app.config['CAP_VIDEO'] = None


from flask_server.api.route import detect
app.register_blueprint(detect)

