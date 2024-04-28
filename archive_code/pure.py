from ultralytics import YOLO
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')

# load video
video_path = 'videos/people.mp4'
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True, classes=0, tracker="botsort.yaml")

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):  # Exit on 'q'
            break
        elif key == ord('p') or key == ord('P'):  # Pause on 'p'
            while True:
                key = cv2.waitKey(1)
                if key == ord('p') or key == ord('q') or key == ord('Q') or key == ord('p'):
                    break
            if key == ord('q') or key == ord('Q'):
                break