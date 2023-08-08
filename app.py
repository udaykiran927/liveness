from flask import Flask,render_template,redirect,request,Response,make_response,url_for,session,jsonify
import cv2
import os
import json
import urllib.request
from datetime import datetime
import requests
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from io import StringIO
import pyrebase
import base64
import time
import io

app=Flask(__name__)

config={
    "apiKey": "AIzaSyAYey8JOEz4XrP_kZTFV0KSwIU9QK8FmCo",
  "authDomain": "mits-students-data.firebaseapp.com",
  "projectId": "mits-students-data",
  "databaseURL":"https://mits-students-data-default-rtdb.firebaseio.com/",
  "storageBucket": "mits-students-data.appspot.com",
  "messagingSenderId": "25326053045",
  "appId": "1:25326053045:web:2416e76031cbf2a78bc0fc",
  "measurementId": "G-8Q4Q08D5T5"
}
firebase=pyrebase.initialize_app(config)
database=firebase.database()
storage=firebase.storage()

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/capture",methods=["POST","GET"])
def capture():
    captured_image = request.form['image']
    roll= request.form.get('rollno').upper()
    encoded_data = captured_image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img1 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = Image.fromarray(img1)
    save_path = f"upload_images/{roll}.jpg"
    image.save(save_path)
    storage.child(f'{roll}_cap.jpg').put(save_path)
    os.remove(save_path)
    '''captured_image = request.form['image']
    roll_no=request.form.get('rollno').upper()
    year=request.form.get('year')
    dept=request.form.get('dept').upper()
    encoded_data = captured_image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    #img1= cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    save_path = "upload_images/"+f'{roll}.jpg'
    cv2.imwrite(save_path,nparr)
    storage.child(f'{roll}_cap.jpg').put(save_path)
    os.remove(save_path)'''
    return render_template("index.html",msg="captured")
    

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')

