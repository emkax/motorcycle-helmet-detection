import requests
import cv2
import live_model_rcnn
import time
import datetime


stream_url = ["https://cctv.balitower.co.id/Tomang-004-702108_2/index.fmp4.m3u8","TOMANG"] #TOMANG
# stream_url = ["https://cctv.balitower.co.id/Gelora-017-700470_9/index.fmp4.m3u8","GBK"] #GBK

cap = cv2.VideoCapture(stream_url[0])

cv2.namedWindow("liveview",cv2.WINDOW_NORMAL)
cv2.resizeWindow("liveview",640,640)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        cur_time = str(datetime.datetime.now()).split(" ")[1].replace(":", "_")
        filename = f"./faster_rcnn_captured/{cur_time}.png"
        time.sleep(0.1)
        # Baca frame
        ret, frame = cap.read()
        if stream_url[1] == "GBK":
            pass
            # frame = frame[600:1200, 480:1520] #posisis jalanan
        elif stream_url[1] == "TOMANG":
            pass
            # frame = frame[200:700, 150:1620] #posisi jalanan 
        
        if ret:
            live_model_rcnn.predict(frame)
            cv2.imshow("liveview", frame)


            if cv2.waitKey(1) & 0xFF == ord('l'):
                cv2.imwrite(filename,frame)
                # skip_iter = 6
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Failed to read a frame from the stream.")
            pass

    cap.release()
    cv2.destroyAllWindows()
