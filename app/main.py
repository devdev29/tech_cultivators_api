import base64
import io
import numpy
from flask import Flask, request
import flask
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from PIL import Image
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)
CORS(app)

def get_from_b64(b64_string):
    face_bytes = bytes(b64_string, 'utf-8')
    face_bytes = face_bytes[face_bytes.find(b'/9'):]
    im = Image.open(io.BytesIO(base64.b64decode(face_bytes)))
    im = im.resize((128,128))
    im=numpy.array(im).astype(numpy.float32)
    im=im/255.0
    im=numpy.expand_dims(im, axis=0)
    return im

@app.route('/bazaar_bhav',methods=['GET'])
def get_bhaav():
        bazaar_page=requests.get('https://msamb.com/').content
        souped_page=BeautifulSoup(bazaar_page, 'html.parser')
        final_bhaav={'commodity':[], 'qty':[], 'price':[]}
        bhaav_html=souped_page.find_all(name='tr')[1:]

        for bhaav in bhaav_html:
                final_bhaav['commodity'].append(bhaav.find_all('td')[0].text)
                final_bhaav['qty'].append(bhaav.find_all('td')[1].text)
                final_bhaav['price'].append(bhaav.find_all('td')[3].text)
        
        return flask.jsonify(final_bhaav)

@app.route('/plant_disease',methods=['GET','POST'])
def detect_disease():
        b64_image=request.get_json()
        in_image=get_from_b64(b64_image)
        model=Interpreter('./app/model/beta_plant_disease.tflite')
        model.allocate_tensors()

        input_details=model.get_input_details()
        output_details=model.get_output_details()

        model.set_tensor(input_details[0]['index'], in_image)
        model.invoke()

        prediction=model.get_tensor(output_details[0]['index'])

        with open('../labels.txt') as lfile:
                for line in lfile.readlines():
                        if prediction.argmax() in line.split(' '):
                                return line.split(' ')[1]

in_image= Image.open('C:/Users/avita/ag_models/plant_disease_detection/train/Corn___Common_rust/image (4).JPG')
model=Interpreter('./app/model/beta_plant_disease.tflite')
in_image = in_image.resize((128,128))
in_image=numpy.array(in_image).astype(numpy.float32)
in_image=in_image/255.0
in_image=numpy.expand_dims(in_image, axis=0)
model.allocate_tensors()

input_details=model.get_input_details()
output_details=model.get_output_details()

model.set_tensor(input_details[0]['index'], in_image)
model.invoke()

prediction=model.get_tensor(output_details[0]['index'])
with open('./labels.txt') as lfile:
        for line in lfile.readlines():
                if prediction.argmax() == int(line.split(' ')[0]):
                        print(line.split(' ')[1])