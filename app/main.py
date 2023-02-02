import base64
import io
import numpy
from flask import Flask, request
import flask
from flask_cors import CORS
from PIL import Image
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)
CORS(app)

def get_from_b64(b64_string):
    b64_string=str(b64_string)    
    face_bytes = bytes(b64_string, 'utf-8')
    face_bytes = face_bytes[face_bytes.find(b'/9'):]
    im = Image.open(io.BytesIO(base64.b64decode(face_bytes)))
    im = im.resize((128,128))
    im=numpy.array(im).astype(numpy.float32)
    im=im/255.0
    im=numpy.expand_dims(im, axis=0)
    return im

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
        with open('./labels.txt') as lfile:
                for line in lfile.readlines():
                        if prediction.argmax() == int(line.split()[0]):
                                resp={'plant':line.split()[1], 'disease':line.split()[2]}
        resp=flask.jsonify(resp)
        resp.headers.add('Access-Control-Allow-Origin', '*')
        return resp

if __name__=='__main__':
        app.run()
