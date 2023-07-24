from flask import Flask,render_template,request,Response,jsonify
import cv2
import urllib
from pathlib import Path
import numpy as np
import onnxruntime
import progressbar
from PIL import Image
from torchvision import transforms as T
import os
import requests
from facetools import LivenessDetection
import base64
import time
import io


app=Flask(__name__)

facebank_path = "/static/reynolds.csv"
deepPix_checkpoint_path = "/static/OULU_Protocol_2_model_0_0.onnx"
livenessDetector = Livenessdetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/capture",methods=["POST","GET"])
def capture():
    captured_image = request.form['image']

    encoded_data = captured_image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img1= cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img1.shape[:2])
    def extract_face(image, bbox):
        # Unpack the bounding box coordinates
        x_min, y_min, x_max, y_max = bbox

        # Crop the face region from the original image
        face = image[y_min:y_max, x_min:x_max]

        return face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale (face detection works on grayscale images)
    gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(faces)
    (x,y,w,h)=faces[0]
    bbox = [x, y, x + w, y + h]

    # Extract the face region using the bounding box and add it to the list
    face_arr = extract_face(img1, bbox)
    liveness_score=livenessDetector(face_arr)
    print(liveness_score)
    if liveness_score>0.65:
        return "Real"
    else:
        return "fake"
    #return str(face_arr)

if __name__=='__main__':
    app.run(debug=True)

