import csv
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

video_path = "videos/people.mp4"
cap = cv2.VideoCapture(video_path)

# Track history and data storage
track_history = defaultdict(lambda: [])
track_data = defaultdict(list)

frame_number = 0  # Initialize frame counter

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True, classes=0)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        annotated_frame = results[0].plot('''conf=False, labels=False''')

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_x, center_y = float(x + w / 2), float(y + h / 2)  # Calculate center x and y
            
            track = track_history[track_id]
            track.append((center_x, center_y))  # Append current center point to track

            if len(track) >= 2:
                # Calculate distance and slope between the last two points
                dx = track[-1][0] - track[-2][0]  # Difference in x
                dy = track[-1][1] - track[-2][1]  # Difference in y
                distance = np.sqrt(dx**2 + dy**2)  # Euclidean distance
                
                slope = dy / dx if dx != 0 else float('inf')  # Calculate slope, handling division by zero
                
                print(f"ID {track_id}: Distance = {distance:.2f} pixels, Slope = {slope:.2f}")
                track_data[track_id].append((frame_number, center_x, center_y, distance, slope))

            if len(track) > 50:  # Optional: Limit the size of the track history
                track.pop(0)

        # Display frame and handle keypresses as before
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        frame_number += 1

        # Exit conditions
        if key == ord('q'):
            break
        elif key == ord('p'):
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


# Write track data to CSV file
with open('chen_track_data_3.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Case_ID', 'Frame_Number', 'X', 'Y', 'Distance_Traveled', 'slope'])
    for track_id, data in track_data.items():
        for frame_info in data:
            frame_number, x, y, distance, speed = frame_info
            writer.writerow([track_id] + list(frame_info))

