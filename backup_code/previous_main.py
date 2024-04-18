import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Function to detect anomalies in tracks
def detect_anomalies(track_history, frame_number):
    # Constants for anomaly detection (placeholders, define these as per your application)
    MAX_SPEED_THRESHOLD = 3.5
    MIN_SPEED_THRESHOLD = 0.6

    for track_id, track in track_history.items():
        # 速度異常
        if len(track) >= 2:
            speeds = [np.sqrt((track[i][0]-track[i-1][0])**2 + (track[i][1]-track[i-1][1])**2) for i in range(1, len(track))]
            avg_speed = np.mean(speeds)
            
            print("length of speeds = ", len(speeds))
            print("length of track_history = ", len(track_history))
            print("avg_speed = ", avg_speed, "\n\n")
            
            if avg_speed > MAX_SPEED_THRESHOLD :
                # anomalies[track_id].append(f"Speed anomaly detected at frame {frame_number}. Avg speed: {avg_speed:.2f}")
                anomaly_point = track[-1]
                # cv2.putText(annotated_frame, "FAST", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # print("速度異常")
            if avg_speed < MIN_SPEED_THRESHOLD:
                anomaly_point = track[-1]
                # cv2.putText(annotated_frame, "SLOW", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')

video_path = "videos/people.mp4"
cap = cv2.VideoCapture(video_path)

# Track history and anomaly data storage
track_history = defaultdict(lambda: [])

# print("track_history = ", track_history, "\n")

# track_data = defaultdict(list)
anomalies_detected = defaultdict(list)

frame_number = 0  # Initialize frame counter

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, classes=0)
        # print("results = ", results, "\n")
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist() #track_ids =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28] 
        # print("track_ids = ", track_ids, "\n")
        annotated_frame = results[0].plot(conf=False)
        # print("track_history = ", track_history, "\n")
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            # center_x, center_y = float(x), float(y + h / 2)  # Calculate center x and y
            
            track = track_history[track_id]
            track.append((x, y))  # Append current center point to track

            if len(track) > 50:  # Limit track history size
                track.pop(0)

            # Draw walking paths
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)

        # Detect anomalies based on the current track history
        detect_anomalies(track_history, frame_number)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        frame_number += 1

        if key == ord('q'):  # Exit on 'q'
            break
        elif key == ord('p'):  # Pause on 'p'
            while True:
                key = cv2.waitKey(1)
                if key == ord('p') or key == ord('q'):
                    break
            if key == ord('q'):
                break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# # Optionally, print or log the detected anomalies
# for track_id, anomaly_messages in anomalies_detected.items():
#     for message in anomaly_messages:
#         print(f"Track ID {track_id}: {message}")
