# -*- coding: utf-8 -*-

from flask import Flask, jsonify;
from flask import request;
from flask_script import Manager;
import os, time, base64, platform
import mydetect

app = Flask(__name__)
manager = Manager(app)

version_str = "v0.01"

def getWorkpath():
    if (platform.system()=="Windows"):
        return 'd:/workroom/testroom/'
    else:
        return '/home/leon/workroom/yolov3/video/'

@app.route('/')
def index():
    hi_str = '<h1>Hi, yolov3 user\n</h1>'
    return '%s%s' % (hi_str, version_str)

@app.route('/yolov3', methods=['POST'])
def yolov3_func():
    res = {}
    req = request.form
    format = req.get("format", "jpg")
    image_base64_str = req.get("image", None)
    if (image_base64_str is None):
        res['code'] = 1001
        res['message'] = "no image"
        resp = jsonify(res)
        resp.status_code = 200
        return resp
    localfile = getWorkpath() + str(int(time.time() * 1000)) + "." + format
    print(localfile)
    image_bytes = base64.b64decode(image_base64_str)
    with open(localfile, "wb") as wfp:
        wfp.write(image_bytes)
    objects = mydetect.mydetect_test(localfile)
    res['code'] = 0
    res['message'] = ""
    res['objects'] = objects
    resp = jsonify(res)
    resp.status_code = 200
    return resp

if __name__=='__main__':
    print('Hi, this is yolov3 service program')
    manager.run()
