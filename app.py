from flask import Flask,render_template,redirect,request,Response,make_response,url_for,session,jsonify
import cv2
import os
import json
import urllib.request
from datetime import datetime
import requests
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from torchvision import transforms as T
from pathlib import Path
import pandas as pd
from io import StringIO
from typing import List, Tuple
import mediapipe as mp
import base64
import time
import io

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def get_size(img):
    if isinstance(img, (np.ndarray, torch.Tensor)):
        return img.shape[1::-1]
    else:
        return img.size


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = cv2.resize(
            img, (image_size, image_size), interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        img = img[box[1] : box[3], box[0] : box[2]]
        out = (
            imresample(
                img.permute(2, 0, 1).unsqueeze(0).float(), (image_size, image_size)
            )
            .byte()
            .squeeze(0)
            .permute(1, 2, 0)
        )
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def extract_face(img, box, image_size=160, margin=0):
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)
    return face
    
class FaceDetection:
    def __init__(self, max_num_faces: int = 1):
        self.detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces, static_image_mode=True
        )

    def __call__(self, image) -> Tuple[List[np.ndarray], List[List[int]]]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        boxes = []
        faces = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array(
                    [(pt.x * w, pt.y * h) for pt in prediction.landmark],
                    dtype=np.float64,
                )
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                face_arr = extract_face(image, bbox.flatten().tolist())
                boxes.append(bbox)
                faces.append(face_arr)
        return faces, boxes

faceDetector = FaceDetection(max_num_faces=1)

app=Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/capture",methods=["POST","GET"])
def capture():
    captured_image = request.form['image']
    encoded_data = captured_image.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img1= cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces, boxes = faceDetector(img1)
    return render_template("index.html",msg=str(faces))
    

if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0')

