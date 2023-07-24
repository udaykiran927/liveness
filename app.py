from flask import Flask,render_template,redirect,request,Response,make_response,url_for,session,jsonify
import cv2
import os
import json
from datetime import datetime
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from facetools import FaceDetection, LivenessDetection
import pandas as pd
import pyrebase
import firebase_admin
from firebase_admin import credentials, storage
import pickle
import base64
import time
import io


app=Flask(__name__)

root = Path(os.path.abspath(__file__)).parent.absolute()
data_folder = root / "data"

#resNet_checkpoint_path = data_folder / "checkpoints" / "InceptionResnetV1_vggface2.onnx"
facebank_path = data_folder / "reynolds.csv"

deepPix_checkpoint_path = data_folder / "checkpoints" / "OULU_Protocol_2_model_0_0.onnx"

faceDetector = FaceDetection(max_num_faces=1)
livenessDetector = LivenessDetection(checkpoint_path=deepPix_checkpoint_path.as_posix())

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/capture",methods=['POST','GET'])

@app.route("/capture",methods=["POST","GET"])
def capture():
    captured_image = request.form['image']
    '''if year=="1":
        std_data=database.child("First").child(dept).child(roll_no).get()
    elif year=="2":
        std_data=database.child("Second").child(dept).child(roll_no).get()
    elif year=="3":
        std_data=database.child("Third").child(dept).child(roll_no).get()
    elif year=="4":
        std_data=database.child("Four").child(dept).child(roll_no).get()
    if std_data.val()['Room']=="":
            return render_template("student.html",msg="No Class Scheduled Yet.")'''

    encoded_data = captured_image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img1= cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img1.shape[:2])
    '''image1= Image.fromarray(img1)
    image_bytes = io.BytesIO()
    image1.save(image_bytes, format="JPG")
    image_bytes = image_bytes.getvalue()
    roll_path=storage.child(f"{roll_no}.jpg").get_url(None)
    response = requests.get(roll_path)'''
    '''faces, boxes = faceDetector(img1)
    print(faces)
    print(boxes)'''
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
    '''for face_arr in :
        #min_sim_score, mean_sim_score = identityChecker(face_arr)
        liveness_score = livenessDetector(face_arr)
        if liveness_score>0.65:
            print("real")
            return "Real"
        else:
            return render_template("index.html",msg="Fake...Don't Cheat us 😄")'''
    liveness_score=livenessDetector(face_arr)
    print(liveness_score)
    return str(liveness_score)

if __name__=='__main__':
    app.run(debug=True)

