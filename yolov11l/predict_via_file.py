from ultralytics import YOLO

import cv2

import datetime

model = YOLO("./demonstration/full_custom_trained_yolov11.pt") #YOLOv11 large model


def predict(frame):
    results = model.predict(frame,conf=0.25)

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

                # if names[i] == "no helmet":
                #     cv2.imwrite(file_path,frame)


        except Exception as e:
            print(e)
            print("got no detection")


import os 
import random

path = './tomang_dataset/images/'

dir = os.listdir(path)

save_folder = './tomang_dataset/saved_predicted'

for file in dir:
    if file[len(file)-3:len(file)] == 'png':
        frame = cv2.imread(os.path.join(path,file))

        predict(frame)

        cv2.namedWindow("liveview",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("liveview",640,640)

        cv2.imshow("liveview",frame)

        key = cv2.waitKey(0) & 0xFF  # wait for key press

        if key == ord('c'):  # if 'c' is pressed
            save_path = os.path.join(save_folder, file)
            
            cv2.imwrite(save_path, frame)
            print(f"Saved: {save_path}")

        cv2.destroyWindow("liveview")

    else:
        continue


