import cv2
import model_predict_all
import time
import threading

# List of streams and their names
stream_urls = [
    ("https://cctv.balitower.co.id/Tomang-004-702108_2/index.fmp4.m3u8", "TOMANG")
    # ("https://another-stream-url.com", "GBK")
]

def get_predicted_frame(frame, model_name):
    frame_copy = frame.copy()
    model_predict_all.predict(frame_copy, model_name, 0.25)
    return frame_copy

def capture_and_process(stream_url, base_name):
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print(f"Error: Could not open video stream {stream_url}")
        return

    while True:
        time.sleep(0.1)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error: Failed to read a frame from {stream_url}")
            continue

        h, w = 640, 640
        frame_resized = cv2.resize(frame, (w, h))

        output1 = [None]
        output2 = [None]

        def run_model(model_name, output_ref):
            output_ref[0] = get_predicted_frame(frame_resized, model_name)

        t1 = threading.Thread(target=run_model, args=("yolov8l", output1))
        t2 = threading.Thread(target=run_model, args=("yolov11l", output2))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Stack side-by-side horizontally
        if output1[0] is not None and output2[0] is not None:
            combined = cv2.hconcat([output1[0], output2[0]])
        elif output1[0] is not None:
            combined = output1[0]
        elif output2[0] is not None:
            combined = output2[0]
        else:
            continue

        cv2.imshow(f"{base_name} - Combined", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    threads = []
    for stream_url, name in stream_urls:
        t = threading.Thread(target=capture_and_process, args=(stream_url, name))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()