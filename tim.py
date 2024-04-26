import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import statistics

MAX_SPEED_THRESHOLD = 0
MIN_SPEED_THRESHOLD = 0
NORMAL_SPEED = 0
DIRECTION_CHANGE_THRESHOLD = 150

TOO_FAST = defaultdict(lambda: False)
TOO_SLOW = defaultdict(lambda: False)
STOP = defaultdict(lambda: False)
DIRECTION = defaultdict(lambda: 0)
DIRECTION_TIME= defaultdict(lambda: 0)
SLOW_TIME = defaultdict(lambda: 0)
STOP_TIME = defaultdict(lambda: 0)
FAST_TIME = defaultdict(lambda: 0)

LAST_UPDATE_FRAME = defaultdict(lambda: 0)
def Swap(a,b):
    if a>=b:
        return a,b
    else:
        return b,a
def mean_and_sigma(track_history, frame_number):
    everyones_speed = []
    # Calculation
    for track_id, track in track_history.items():
        if len(track) >= 10:
            speeds = [np.sqrt((track[-1][0]-track[-10][0])**2 + (track[-1][1]-track[-10][1])**2) for i in range(1, len(track))]
            individual_avg_speed = np.mean(speeds)
            if not np.isnan(individual_avg_speed):
                everyones_speed.append(individual_avg_speed)
        
    print("everyones_speed = ", everyones_speed)
    
    everyones_avg = np.mean(everyones_speed) 
    speed_std = np.std(everyones_speed)
    
    print("everyones_avg = ", everyones_avg)
    print("speed_std = ", speed_std)
    
    return (everyones_avg/90), (everyones_avg + 2*speed_std), (everyones_avg - 1.5*speed_std)
    
    
# Function to detect anomalies in tracks
def detect_anomalies(track_history, frame_number, annotated_frame):
    global MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD, NORMAL_SPEED,DIRECTION_CHANGE_THRESHOLD
    global TOO_FAST, TOO_SLOW,STOP,DIRECTION,DIRECTION_TIME,SLOW_TIME,FAST_TIME,STOP_TIME
    global LAST_UPDATE_FRAME
    
    if (frame_number >= 50):     
        if (frame_number % 50 == 0):
            NORMAL_SPEED, MAX_SPEED_THRESHOLD, MIN_SPEED_THRESHOLD = mean_and_sigma(track_history, frame_number)

        for track_id, track in track_history.items():
            
            if((frame_number - LAST_UPDATE_FRAME[track_id]) > 5):
                TOO_FAST.pop(track_id, None)
                TOO_SLOW.pop(track_id, None)
                STOP.pop(track_id, None)
            if((frame_number - LAST_UPDATE_FRAME[track_id]) > 50):
                track_history[track_id] = [(0, 0)]
            ###################################停止###################################
            # if len(track) >= 5:
            # # if len(track) % 5 == 0:
            #     speeds = [np.sqrt((track[-1][0] - track[-5][0])**2 + (track[-1][1] - track[-5][1])**2)]
            #     if len(track) % 50 == 0:
            #         if speeds <= NORMAL_SPEED:
            #             STOP[track_id] = 15
            #             DIRECTION[track_id] = 0
            
            ###################################速度異常(快慢) & 停止###################################
            if len(track) >= 10:
                speeds = [np.sqrt((track[-1][0]-track[-10][0])**2 + (track[-1][1]-track[-10][1])**2)]
                avg_speed = np.mean(speeds)
                
                if (frame_number % 10 == 0):
                    if((avg_speed < MAX_SPEED_THRESHOLD) and (avg_speed > MIN_SPEED_THRESHOLD)):
                        TOO_FAST[track_id] = False
                        TOO_SLOW[track_id] = False

                    if avg_speed > MAX_SPEED_THRESHOLD and FAST_TIME[track_id] < 10:
                        FAST_TIME[track_id]+=1.5
                    else:
                        FAST_TIME[track_id]-=1
                        if FAST_TIME[track_id] < 0:
                            FAST_TIME[track_id] = 0

                    if avg_speed < MIN_SPEED_THRESHOLD:
                        if avg_speed < MIN_SPEED_THRESHOLD*0.75:
                            STOP_TIME[track_id]+=1
                        else:
                            SLOW_TIME[track_id]+=1

                if FAST_TIME[track_id] > 3:
                    TOO_FAST[track_id] = True
                else:
                    TOO_FAST[track_id] = False
                
                if (SLOW_TIME[track_id]+STOP_TIME[track_id] >= 7):
                    TOO_SLOW[track_id] = False
                    STOP[track_id] = False
                    if  SLOW_TIME[track_id] >= STOP_TIME[track_id]:
                        TOO_SLOW[track_id] = True
                        SLOW_TIME[track_id] = 0
                        STOP_TIME[track_id] = 0
                    elif STOP_TIME[track_id] > SLOW_TIME[track_id]:
                        STOP[track_id] = True
                        SLOW_TIME[track_id] = 0
                        STOP_TIME[track_id] = 0

                if TOO_FAST[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "FAST", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2) 
                if TOO_SLOW[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "SLOW", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if STOP[track_id]:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "STOP", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    DIRECTION[track_id] = 0
                    
            
            
            ###################################方向異常###################################
            deviation = 0.3
            errorThreshold = 5
            if len(track) % 10 == 0 and (STOP[track_id] == False or STOP[track_id] == None):
                Xmax,Xmin = Swap(track[-5][1]+(track[-6][1]-track[-10][1])-(track[-10][1]-track[-6][1])*deviation,track[-5][1]+(track[-6][1]-track[-10][1])+(track[-10][1]-track[-6][1])*deviation)
                Ymax,Ymin = Swap(track[-5][0]+(track[-6][0]-track[-10][0])-(track[-10][0]-track[-6][0])*deviation,track[-5][0]+(track[-6][0]-track[-10][0])+(track[-10][0]-track[-6][0])*deviation)

                if ((track[-1][1]) < Xmin or (track[-1][1])>Xmax):
                    DIRECTION_TIME[track_id]+=1
                elif ((track[-1][0]) < Ymin or (track[-1][0]) > Ymax):
                    DIRECTION_TIME[track_id]+=1

                if len(track) % 50 == 0 and DIRECTION_TIME[track_id] > 1:
                    DIRECTION_TIME[track_id]-=1
                if DIRECTION_TIME[track_id] > errorThreshold:
                    DIRECTION[track_id] = 20
                if DIRECTION[track_id] > 0 and STOP[track_id] == False:
                    anomaly_point = track[-1]
                    cv2.putText(annotated_frame, "DIRECTION", (int(anomaly_point[0]), int(anomaly_point[1]+30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    DIRECTION[track_id]-=1
                if((frame_number - LAST_UPDATE_FRAME[track_id]) > 5):
                    DIRECTION.pop(track_id, None)
                    DIRECTION[track_id] = 0
            ###################################方向異常###################################
            

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
        results = model.track(frame, persist=True, classes=0, tracker="botsort.yaml")
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot(conf=False, labels=False)
       
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
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()