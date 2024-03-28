import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Function to detect anomalies in tracks
def detect_anomalies(track_history, frame_number):
    anomalies = defaultdict(list)
    # Constants for anomaly detection (placeholders, define these as per your application)
    MAX_SPEED_THRESHOLD = 50
    MIN_SPEED_THRESHOLD = 5
    STOP_SPEED_THRESHOLD = 2
    DIRECTION_CHANGE_THRESHOLD = 45
    REVERSE_ANGLE_THRESHOLD = 160

    for track_id, track in track_history.items():
        # 速度異常
        if len(track) >= 2:
            speeds = [np.sqrt((track[i][0]-track[i-1][0])**2 + (track[i][1]-track[i-1][1])**2) for i in range(1, len(track))]
            avg_speed = np.mean(speeds)
            if avg_speed > MAX_SPEED_THRESHOLD or avg_speed < MIN_SPEED_THRESHOLD:
                anomalies[track_id].append(f"Speed anomaly detected at frame {frame_number}. Avg speed: {avg_speed:.2f}")
                # print("速度異常")
        # 方向突變
        if len(track) >= 3:
            direction_changes = []
            for i in range(2, len(track)):
                angle1 = np.degrees(np.arctan2(track[i-1][1]-track[i-2][1], track[i-1][0]-track[i-2][0]))
                angle2 = np.degrees(np.arctan2(track[i][1]-track[i-1][1], track[i][0]-track[i-1][0]))
                angle_change = np.abs(angle1 - angle2)
                if angle_change > DIRECTION_CHANGE_THRESHOLD:
                    direction_changes.append(frame_number)
            if direction_changes:
                anomalies[track_id].append(f"Direction change anomaly detected at frames: {direction_changes}")
        #停止
        if len(track) >= 2:
            speeds = [np.sqrt((track[i][0]-track[i-1][0])**2 + (track[i][1]-track[i-1][1])**2) for i in range(1, len(track))]
            stop_frames = [i for i, speed in enumerate(speeds, start=1) if speed < STOP_SPEED_THRESHOLD]
            if stop_frames:
                anomalies[track_id].append(f"Stop movement detected at frames: {stop_frames}")
        
        # 反向移動
        if len(track) >= 4:
            for i in range(3, len(track)):
                angle1 = np.degrees(np.arctan2(track[i-2][1]-track[i-3][1], track[i-2][0]-track[i-3][0]))
                angle2 = np.degrees(np.arctan2(track[i][1]-track[i-1][1], track[i][0]-track[i-1][0]))
                angle_change = min(np.abs(angle1 - angle2), 360 - np.abs(angle1 - angle2)) 
                if angle_change > REVERSE_ANGLE_THRESHOLD:
                    anomalies[track_id].append(f"Reverse movement detected at frame {frame_number}")

    return anomalies

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

video_path = "videos/people.mp4"
cap = cv2.VideoCapture(video_path)

# Track history and anomaly data storage
track_history = defaultdict(lambda: [])
track_data = defaultdict(list)
anomalies_detected = defaultdict(list)

frame_number = 0  # Initialize frame counter

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, classes=0)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot(conf=False)

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_x, center_y = float(x), float(y + h / 2)  # Calculate center x and y
            
            track = track_history[track_id]
            track.append((center_x, center_y))  # Append current center point to track

            if len(track) > 50:  # Limit track history size
                track.pop(0)

            # Draw walking paths
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)

        # Detect anomalies based on the current track history
        anomalies = detect_anomalies(track_history, frame_number)
        for track_id, anomaly_messages in anomalies.items():
            anomalies_detected[track_id].extend(anomaly_messages)

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
