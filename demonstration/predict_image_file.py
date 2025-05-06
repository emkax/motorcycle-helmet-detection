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
from PIL import Image

device = cuda.get_current_device()
print(device)

model_type = "custom_yolov8l"

model1 = YOLO("./helmet_base_yolo11l_best.pt") #YOLOv11 large model
model2 = YOLO("./helmet_base_yolo11n_best.pt") #YOLOv11 nano model
model3 = YOLO("./helmet_base_yolov8l_best.pt") #YOLOv8 large model
custom_model_yolov11l = YOLO("./full_custom_trained_yolov11.pt") #pengganti yolov11 large model
cusotm_model_yolov8l = YOLO("./full_trained_custom_model_yolov8l.pt") #pengganti yolov8 large model

img = cv2.imread('./image_pres/test_malam.png')

# cv2.namedWindow("liveview",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("liveview",640,640)

custom_model_recommended_confidence = 0.25


def predict(frame,model_type,confidence_score):
    if model_type == "yolov8l":
        model = model3
    elif model_type == "yolov11n":
        model = model2
    elif model_type == "yolov11l":
        model = model1
    elif model_type == "custom_yolov11l":
        model = custom_model_yolov11l
    elif model_type == "custom_yolov8l":
        model = cusotm_model_yolov8l

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
    print(xywh)
    print(xywhn)
    # print(xyxy)
    # print(xyxyn)
    # print(names)
    # print(confs)

    drawBox(xyxy,frame,names)
    cv2.imwrite(f"C:\\Users\\Michael\\Documents\\python projects\\Karya tulis Computer Vision BHK 2025\\demonstration\\image_pres\\prediction\\{model_type}_malam.png",img)
    # cv2.imshow('liveview',img)
    # cv2.waitKey(0)

def drawBox(xyxy,frame,names):
    for i in range(len(xyxy)):
        try:
            if names[i] == "full-faced" or names[i] == "half-faced" or names[i] == "full-face helmet" or names[i] == "half-face helmet": 
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

predict(img,model_type,0.25)