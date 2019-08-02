# -*- coding: utf-8 -*-
#

import base64
import urllib, http.client
import json, os


def http_request(http_url, reqDict):
    cburlclass = urllib.parse.urlparse(http_url)
    ret = reqDict

    data = None
    params = urllib.parse.urlencode(ret).encode('utf-8')
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    try:
        conn = http.client.HTTPConnection(host=cburlclass.hostname, port=cburlclass.port, timeout=20)
        conn.request("POST", cburlclass.path, params, headers)
        response = conn.getresponse()
        if (response.status==200):
            data = json.loads(response.read().decode('utf-8'))
        else:
            print(response.status, response.reason)
        # print
        #
    except Exception as ex:
        print(str(ex))
        return None
    else:
        conn.close()
    return data


def object_detect(localimage, http_url):
    with open(localimage, "rb") as fp:
        imagedata = fp.read(20 * 1000 * 1000)
    print("imagedata : ", len(imagedata))
    imagedata_base64_str = str(base64.b64encode(imagedata), "utf-8")
    print("imagedata_base64_str : ", len(imagedata_base64_str))
    image_dict = dict()
    image_dict['image'] = imagedata_base64_str
    image_dict['format'] = os.path.splitext(localimage)[1].replace(".", "")
    print(image_dict)
    respDict = http_request(http_url, image_dict)
    if (respDict is not None):
        # print(respDict.get('objects', []))
        return respDict.get('objects', [])
    else:
        print("error...")
        return []

def test():
    print("Hi, this is sound classify test program")

    sc_url = "http://127.0.0.1:1108/yolov3"
    localwav = "D:\\memorial12.jpg"
    ret = object_detect(localwav, sc_url)
    print(ret)


if __name__=="__main__":
    test()

