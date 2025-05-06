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

YOLO_variant = "yolov8l"
if YOLO_variant == "yolov11l":
    model = YOLO("./newest_trained_model/helmet_base_yolo11l_best.pt") #YOLOv11 large model
elif YOLO_variant == "yolov11n":
    model = YOLO("./trained model/helmet_base_yolo11n_best.pt") #YOLOv11 nano model
elif YOLO_variant == "yolov8l":
    model = YOLO("./helmet_base_yolov8l_best.pt") #YOLOv8 large model


def predict(frame):
    results = model.predict(frame,conf=0.1)

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
            if names[i] == "full-faced" or names[i] == "half-faced": 
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