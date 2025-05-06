import shutil
import torch
from ultralytics import YOLO
from roboflow import Roboflow
import random
import shutil
import os
import tensorflow as tf
from numba import cuda
import cv2
import datetime


device = cuda.get_current_device()
print(device)


model1 = YOLO("./full_custom_trained_yolov11.pt") #YOLOv11 large model
model2 = YOLO("./helmet_base_yolo11n_best.pt") #YOLOv11 nano model
model3 = YOLO("./full_trained_custom_model_yolov8l.pt") #YOLOv8 large model


def predict(frame,model_type,confidence_score):
    if model_type == "yolov8l":
        model = model3
    elif model_type == "yolov11n":
        model = model2
    elif model_type == "yolov11l":
        model = model1

    results = model.predict(frame,conf=confidence_score)

    # Access the results
    for result in results:
        xywh = result.boxes.xywh  
        xywhn = result.boxes.xywhn  
        xyxy = result.boxes.xyxy  
        xyxyn = result.boxes.xyxyn  
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  
        confs = result.boxes.conf  
    
    print(names)
    # return xyxy
    # print(xywh)
    # print(xywhn)
    # print(xyxy)
    # print(xyxyn)
    # print(names)
    # print(confs)

    drawBox(xyxy,frame,names)

def drawBox(xyxy,frame,names):
    for i in range(len(xyxy)):
        try:
            if names[i] == "full-faced" or names[i] == "half-faced" or names[i] == "full-face helmet" or names[i] == "half-faced helmet": 
                cv2.rectangle(frame, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (0, 255, 0), 5)
                text = names[i]  
                x, y = int(xyxy[i][0]), int(xyxy[i][1]) - 10  


                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width+5, y + 5), (0, 0, 0), cv2.FILLED)

                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  

            elif names[i] == "no helmet" or names[i] == "invalid":
                cur_time = str(datetime.datetime.now()).split(" ")[1].replace(":", "_") 
                file_path = f"./nohelmetdetect/{cur_time}.png"

                cv2.rectangle(frame, (int(xyxy[i][0]), int(xyxy[i][1])), (int(xyxy[i][2]), int(xyxy[i][3])), (255, 0, 0), 5)
                text = names[i]  
                x, y = int(xyxy[i][0]), int(xyxy[i][1]) - 10 

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width+5, y + 5), (0, 0, 0), cv2.FILLED)

                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)  

                if names[i] == "no helmet":
                    cv2.imwrite(file_path,frame)


        except Exception as e:
            print(e)
            print("got no detection")