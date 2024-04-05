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
        if len(track) >= 11:
            speeds = [np.sqrt((track[-1][0]-track[-11][0])**2 + (track[-1][1]-track[-11][1])**2) for i in range(1, len(track))]
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
            if len(track) >= 11:
                speeds = [np.sqrt((track[-1][0]-track[-11][0])**2 + (track[-1][1]-track[-11][1])**2)]
                avg_speed = np.mean(speeds)
                
                if (frame_number % 50 == 0):
                    TOO_FAST[track_id] = avg_speed > MAX_SPEED_THRESHOLD
                    TOO_SLOW[track_id] = avg_speed < MIN_SPEED_THRESHOLD
                    if((avg_speed < MAX_SPEED_THRESHOLD) and (avg_speed > MIN_SPEED_THRESHOLD)):
                        TOO_FAST[track_id] = False
                        TOO_SLOW[track_id] = False
                        
                if((frame_number - LAST_UPDATE_FRAME[track_id]) > 5):
                    TOO_FAST.pop(track_id, None)
                    TOO_SLOW.pop(track_id, None)
                                    
                    
                if TOO_FAST[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "FAST", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                if TOO_SLOW[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "SLOW", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

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
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot(conf=False)
       
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                footpoint_x, footpoint_y = float(x), float(y)
                # footpoint_x, footpoint_y = float(x), float(y + h / 2)  
                track = track_history[track_id]
                track.append((footpoint_x, footpoint_y))
                LAST_UPDATE_FRAME[track_id] = frame_number
                
                if len(track) > 50:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)
                
                detect_anomalies(track_history, frame_number, annotated_frame)

                cv2.putText(annotated_frame, str(frame_number), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            print("No objects detected in this frame.")
            cv2.putText(frame, str(frame_number), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            img_height, img_width, img_channels = frame.shape 
            cv2.putText(frame, "No people detected", (int(img_height/3),int(img_width/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Show the original frame when no objects are detected
            cv2.imshow("YOLOv8 Tracking", frame)

        frame_number += 1
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
        # elif key == ord('r'):
        #      video_path = "videos/self.mp4"
        #      cap = cv2.VideoCapture(video_path)
        # elif key == ord('s'):
        #     video_path = 0
        #     cap = cv2.VideoCapture(video_path)
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
