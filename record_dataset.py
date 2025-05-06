import requests
import cv2
import time
import datetime


stream_url = "https://cctv.balitower.co.id/Tomang-004-702108_2/index.fmp4.m3u8" #TOMANG
# stream_url = "https://cctv.balitower.co.id/Gelora-017-700470_9/index.fmp4.m3u8" #GBK


cap = cv2.VideoCapture(stream_url)
cv2.namedWindow("liveview",cv2.WINDOW_NORMAL)
cv2.resizeWindow("liveview",640,640)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        time.sleep(2)
        ret,frame = cap.read()
        if ret:
            cur_time = str(datetime.datetime.now()).split(" ")[1].replace(":", "_")  
            file_path = f"./custom_dataset/dataset/{cur_time}.png"
            cv2.imwrite(file_path,frame)
            print(f'Successfully saved {file_path}')
            cv2.imshow("liveview",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("failed")
            break

   
    cap.release()
    cv2.destroyAllWindows()
