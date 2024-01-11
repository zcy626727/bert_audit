from flask import Flask, request
import text_classifier
import Image_classifier
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
text_model = text_classifier.TextClassifier()
image_model = Image_classifier.Imageclassifier()


@app.route('/checkText', methods=['GET'])
def checkText():
    text = request.values.get("text")
    if text == None:
        return "输入text参数", 400
    return {
        'access': ~text_model.predict(text=text)[0] & 1
    }

@app.route('/checkImage', methods=['POST'])
def checkImage():
    image = request.files["image"]
    if not image:
        return '没有上传文件', 400
    # image.save("images/"+image.filename)
    nparr = np.frombuffer(image.read(), np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_model.upload(img_np):
        return {
            'access': 1
        }
    else:
        return {
            'access': 0
        }

app.run(host='192.168.239.148', port=5000)
