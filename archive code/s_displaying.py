import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import statistics

MAX_SPEED_THRESHOLD = 0
MIN_SPEED_THRESHOLD = 0

TOO_FAST = defaultdict(lambda: False)
TOO_SLOW = defaultdict(lambda: False)

LAST_UPDATE_FRAME = defaultdict(lambda: 0)

def mean_and_sigma(track_history, frame_number):
    everyones_speed = []
    # Calculation
    for track_id, track in track_history.items():
        speeds = [np.sqrt((track[i][0]-track[i-1][0])**2 + (track[i][1]-track[i-1][1])**2) for i in range(1, len(track))]
        individual_avg_speed = np.mean(speeds)
        if not np.isnan(individual_avg_speed):
            everyones_speed.append(individual_avg_speed)
        
    
    print("everyones_speed = ", everyones_speed)
    
    everyones_avg = np.mean(everyones_speed) #checked
    # Standard Deviation
    speed_std = np.std(everyones_speed)
    
    print("everyones_avg = ", everyones_avg)
    print("speed_std = ", speed_std)
    
    return (everyones_avg + speed_std), (everyones_avg - speed_std)
    
    
# Function to detect anomalies in tracks
def detect_anomalies(track_history, frame_number, annotated_frame):
    global MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD
    global TOO_FAST, TOO_SLOW
    global LAST_UPDATE_FRAME
    
    if (frame_number >= 50):     
        if (frame_number % 50 == 0):
            MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD = mean_and_sigma(track_history, frame_number)

        for track_id, track in track_history.items():
            
            # print("Before TOO_FAST: ", TOO_FAST)
            # print("Before TOO_SLOW: ", TOO_SLOW)
            # if((frame_number - LAST_UPDATE_FRAME[track_id]) > 5):
            #     TOO_FAST.pop(track_id, None)
            #     TOO_SLOW.pop(track_id, None)
            #     # track_history.pop(track_id)
            #     print("After TOO_FAST: ", TOO_FAST)
            #     print("After TOO_SLOW: ", TOO_SLOW)
            #     # continue

            
            if len(track) >= 2:
                speeds = [np.sqrt((track[-1][0]-track[-2][0])**2 + (track[-1][1]-track[-2][1])**2)]
                avg_speed = np.mean(speeds)
                
                if (frame_number % 50 == 0):
                    TOO_FAST[track_id] = avg_speed > MAX_SPEED_THRESHOLD
                    TOO_SLOW[track_id] = avg_speed < MIN_SPEED_THRESHOLD
                    if((avg_speed < MAX_SPEED_THRESHOLD) and (avg_speed > MIN_SPEED_THRESHOLD)):
                        TOO_FAST[track_id] = False
                        TOO_SLOW[track_id] = False
                        
                # print("Before TOO_FAST: ", TOO_FAST)
                # print("Before TOO_SLOW: ", TOO_SLOW)
                if((frame_number - LAST_UPDATE_FRAME[track_id]) > 5):
                    TOO_FAST.pop(track_id, None)
                    TOO_SLOW.pop(track_id, None)
                    # track_history.pop(track_id)
                    # print("After TOO_FAST: ", TOO_FAST)
                    # print("After TOO_SLOW: ", TOO_SLOW)
                    # continue
                
                    
                if TOO_FAST[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "FAST", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                if TOO_SLOW[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "SLOW", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

video_path = "videos/self.mp4"
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
            footpoint_x, footpoint_y = float(x), float(y)
            # footpoint_x, footpoint_y = float(x), float(y + h / 2)  # Calculate center x and y
            track = track_history[track_id]
            track.append((footpoint_x, footpoint_y))  # Append current center point to track
            LAST_UPDATE_FRAME[track_id] = frame_number  # Correctly update last update frame
            
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

