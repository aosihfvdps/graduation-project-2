import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import statistics

MAX_SPEED_THRESHOLD = 0
MIN_SPEED_THRESHOLD = 0

def mean_and_sigma(track_history, frame_number):
    everyones_speed = []
    # Calculation
    for track_id, track in track_history.items():
        speeds = [np.sqrt((track[i][0]-track[i-10][0])**2 + (track[i][1]-track[i-10][1])**2) for i in range(1, len(track))]
        # print("frame_number = ", frame_number)
        # print("track_id = ", track_id)
        # print("speeds = ", speeds)
        individual_avg_speed = np.mean(speeds)
        everyones_speed.append(individual_avg_speed)
    
    everyones_avg = np.mean(everyones_speed) #checked
    # print("everyones_speed = ", everyones_speed)
    # print("everyones_avg = ", everyones_avg)
    
    # Standard Deviation
    speed_std = np.std(everyones_speed)
    # print("everyones_avg + speed_std = ", everyones_avg + speed_std)
    # print("everyones_avg - speed_std = ", everyones_avg - speed_std)
    
    # return (everyones_avg + 5*speed_std), (everyones_avg - speed_std)
    return 50, (everyones_avg - speed_std)
    # return 50, 0.6
    
    

# Function to detect anomalies in tracks
def detect_anomalies(track_history, frame_number, annotated_frame):
    # Constants for anomaly detection (placeholders, define these as per your application)
    # print("frame_number = ", frame_number, "evaluation = ", (MAX_SPEED_THRESHOLD or MIN_SPEED_THRESHOLD) or (frame_number % 50 == 0))
    global MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD
    if (frame_number > 50):     
        if (frame_number % 50 == 0):
            MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD = mean_and_sigma(track_history, frame_number)
        for i in range(50): 
            print("MAX_SPEED_THRESHOLD", MAX_SPEED_THRESHOLD)
            print("MIN_SPEED_THRESHOLD", MIN_SPEED_THRESHOLD)
            
        for track_id, track in track_history.items():
            # 速度異常
            if len(track) >= 11:
                speeds = [np.sqrt((track[-1][0]-track[-11][0])**2 + (track[-1][1]-track[-11][1])**2)]
                avg_speed = np.mean(speeds)
                print("avg_speed", avg_speed)
                if avg_speed > MAX_SPEED_THRESHOLD:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "fast", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                if avg_speed < MIN_SPEED_THRESHOLD:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "slow", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

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
            footpoint_x, footpoint_y = float(x), float(y + h / 2)  # Calculate center x and y
            
            track = track_history[track_id]
            track.append((footpoint_x, footpoint_y))  # Append current center point to track

            if len(track) > 50:  # Limit track history size
                track.pop(0)

            # Draw walking paths
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)

        # Detect anomalies based on the current track history
        detect_anomalies(track_history, frame_number, annotated_frame)

        cv2.putText(annotated_frame, str(frame_number), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # print(frame_number, "\n")
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

