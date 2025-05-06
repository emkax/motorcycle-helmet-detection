import requests
import cv2

stream_url = "https://cctv.balitower.co.id/Bendungan-Hilir-003-700014_2/index.fmp4.m3u8"
# Open the video stream
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        # Read a frame from the stream
        ret, frame = cap.read()
        
        # If the frame was successfully read
        if ret:
            # Display the frame in a window
            cv2.imshow("Livestream", frame)

            # Press 'q' to exit the video window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Failed to read a frame from the stream.")
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
