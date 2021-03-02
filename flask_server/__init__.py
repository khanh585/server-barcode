
from utils.torch_utils import select_device
from models.experimental import attempt_load
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
app.config['UPLOAD_FOLDER'] = r'flask_server\storage\uploads'
app.config['WEIGHT_FILES'] = r'flask_server\weights\last.pt'
app.config['MODEL'] = model
app.config['DEVICE'] = device

from flask_server.api.route import todo
app.register_blueprint(todo)

