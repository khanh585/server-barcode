
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

# create folder
newpath = os.path.join(real_path, r'flask_server\storage\uploads\videos')  
if not os.path.exists(newpath):
    os.makedirs(newpath)

newpath = os.path.join(real_path, r'flask_server\storage\uploads\images')  
if not os.path.exists(newpath):
    os.makedirs(newpath)

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
app.config['AZURE_CONN_STR'] = r"BlobEndpoint=https://papv.blob.core.windows.net/;QueueEndpoint=https://papv.queue.core.windows.net/;FileEndpoint=https://papv.file.core.windows.net/;TableEndpoint=https://papv.table.core.windows.net/;SharedAccessSignature=sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacupx&se=2021-05-31T21:36:09Z&st=2021-04-10T13:36:09Z&spr=https&sig=te5dpk%2FASSegm2WRSVxQZEHX7HIB92MRFuPbv7E70Pg%3D"


from flask_server.api.route import detect
app.register_blueprint(detect)

