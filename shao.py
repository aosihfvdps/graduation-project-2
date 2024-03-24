import csv
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.ensemble import IsolationForest
import pandas as pd

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

video_path = "videos/people.mp4"
cap = cv2.VideoCapture(video_path)

# Track history and data storage
track_history = defaultdict(lambda: [])
track_data = defaultdict(lambda: defaultdict(list))  # Modified to store more structured data

frame_number = 0  # Initialize frame counter

# Initialize Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# 将数据点的数量设置为触发异常检测的阈值
ANOMALY_CHECK_THRESHOLD = 10 

# 用于存储特徵的列表
all_features = []


while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, classes=0)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot(conf=False, labels=False)

        # Prepare data for Isolation Forest
        features = []

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_x, center_y = float(x), float(y + h / 2)
            
            track = track_history[track_id]
            track.append((center_x, center_y))

            if len(track) >= 2:
                dx = track[-1][0] - track[-2][0]
                dy = track[-1][1] - track[-2][1]
                distance = np.sqrt(dx**2 + dy**2)
                slope = dy / dx if dx != 0 else float('inf')

                track_data[track_id]['Frame_Number'].append(frame_number)
                track_data[track_id]['X'].append(center_x)
                track_data[track_id]['Y'].append(center_y)
                track_data[track_id]['Distance_Traveled'].append(distance)
                track_data[track_id]['Slope'].append(slope)

                features.append([distance, slope])

            if len(track) > 50:
                track.pop(0)

        if features:
            all_features.extend(features)# 将新特徵添加到所有特徵的列表中
            if len(all_features) >= ANOMALY_CHECK_THRESHOLD:# 只有当我们有足够的数据点时，才进行异常检测
                # Convert features to DataFrame for Isolation Forest
                df_features = pd.DataFrame(features, columns=['Distance_Traveled', 'Slope'])
                # Fit and predict
                preds = iso_forest.fit_predict(df_features)

            # Mark anomalies
            for idx, pred in enumerate(preds):
                if pred == -1:
                    # Get track ID for the anomaly
                    anomaly_id = list(track_data.keys())[idx]
                    anomaly_point = track_history[anomaly_id][-1]  # Get the latest point
                    cv2.putText(annotated_frame, "WRONG", (int(anomaly_point[0]), int(anomaly_point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        frame_number += 1

        if key == ord('q'):  # Exit on 'q'
            break
        elif key == ord('p'):  # Pause on 'p'
            cv2.waitKey(-1)  # Wait indefinitely until any key is pressed

    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()