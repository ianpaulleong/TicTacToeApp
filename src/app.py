from fastai import *
from fastai.vision import *
import fastai
import yaml
import sys
from io import BytesIO
from typing import List, Dict, Union, ByteString, Any
import os
import flask
from flask import Flask
import requests
import torch
import json
import pickle
import numpy as np
import tttPlayer
from torchvision import transforms

with open("src/config.yaml", 'r') as stream:
    APP_CONFIG = yaml.full_load(stream)

app = Flask(__name__)


#def load_model(path=".", model_name="model.pkl"):
#    learn = load_learner(path, fname=model_name)
#    return learn

def load_model(model_name,path = 'models'):
    with open(path + '/' + model_name,'rb') as handle:
        theModel = pickle.load(handle)
    return theModel


def load_image_url(url: str) -> Image:
    response = requests.get(url)
    img = open_image(BytesIO(response.content))
    return img


def load_image_bytes(raw_bytes: ByteString) -> Image:
    img = open_image(BytesIO(raw_bytes))
    return img


#def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
#    pred_class, pred_idx, outputs = model.predict(img)
#    pred_probs = outputs / sum(outputs)
#    pred_probs = pred_probs.tolist()
#    predictions = []
#    for image_class, output, prob in zip(model.data.classes, outputs.tolist(), pred_probs):
#        output = round(output, 1)
#        prob = round(prob, 2)
#        predictions.append(
#            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
#        )
#
#    predictions = sorted(predictions, key=lambda x: x["output"], reverse=True)
#    predictions = predictions[0:n]
#    return {"class": str(pred_class), "predictions": predictions}
def predict(img, n: int = 3) -> Dict[str, Union[str, List]]:
    imgTensor = img.data
    data_transforms = transforms.Compose([
        transforms.Resize(244),
        #transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    imgTensor = data_transforms(transforms.ToPILImage()(imgTensor))
    imgTensor = imgTensor.unsqueeze(0)
    theReadState = picModel(imgTensor) > 0.45
    theReadState = theReadState.squeeze()
    theStrState = ''
    for ii in range(9):
        jj = ii + 9
        if theReadState[ii] == 1:
            theStrState = theStrState + 'X'
        elif theReadState[jj] == 1:
            theStrState = theStrState + 'O'
        else:
            theStrState = theStrState + ' '
            
        if ii%3 == 2:
            theStrState = theStrState + '\n'
    
    theReadState = theReadState.reshape([6,3])
    theReadStateX = theReadState[0:3,:]
    theReadStateO = theReadState[3:6,:]
    if theReadStateX.sum().item() == theReadStateO.sum().item():
        numpyState = theReadStateX.detach().numpy().astype(int) - theReadStateO.detach().numpy().astype(int)
    else:
        numpyState = theReadStateO.detach().numpy().astype(int) - theReadStateX.detach().numpy().astype(int)
    chosenMove = sheepModel.chooseMove(numpyState)
    chosenMove = float(chosenMove)
    predictions = [{'class': theStrState,'output':'foo','prob':chosenMove/100}]
#    predictions.append(
#            {"class": image_class.replace("_", " "), "output": output, "prob": prob}
#        )
    return {"class": theStrState, "predictions": predictions}


@app.route('/api/classify', methods=['POST', 'GET'])
def upload_file():
    if flask.request.method == 'GET':
        url = flask.request.args.get("url")
        img = load_image_url(url)
    else:
        bytes = flask.request.files['file'].read()
        img = load_image_bytes(bytes)
    res = predict(img)
    return flask.jsonify(res)


#@app.route('/api/classes', methods=['GET'])
#def classes():
#    classes = sorted(model.data.classes)
#    return flask.jsonify(classes)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/config')
def config():
    return flask.jsonify(APP_CONFIG)


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"

    response.cache_control.max_age = 0
    return response


@app.route('/<path:path>')
def static_file(path):
    if ".js" in path or ".css" in path:
        return app.send_static_file(path)
    else:
        return app.send_static_file('index.html')


@app.route('/')
def root():
    return app.send_static_file('index.html')


def before_request():
    app.jinja_env.cache = {}

picModel = load_model('retrainedModel.pickle','models')
sheepModel = load_model('sheepPlayer.pickle','models')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)

    if "prepare" not in sys.argv:
        app.jinja_env.auto_reload = True
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        app.run(debug=False, host='0.0.0.0', port=port)
        # app.run(host='0.0.0.0', port=port)
