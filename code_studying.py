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
        # Object tracking
        results = model.track(frame, persist=True, classes=0)

        # Trajectory updates
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize results
        annotated_frame = results[0].plot('''conf=False, labels=False''')

        # Process each tracked object
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            center_y = y + h / 2
            track.append((float(x), float(center_y)))  # x, y center point
            speed = 0  # Initialize speed
            
            # Calculate distance and speed
            if len(track) >= 2:
                dx, dy = np.diff(track[-2:], axis=0)[0]
                speed = np.sqrt(dx**2 + dy**2)
                
                distance = np.sum(np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1)))
                print(f"ID {track_id}: coordination = ({int(x)},{int(y)}), Distance = {distance:.2f} pixels, speed={speed}")
                track_data[track_id].append((frame_number, x, center_y, distance, speed))

            if len(track) > 50:
                track.pop(0)

            # Draw trajectory line
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 256, 0), thickness=5)

        # Display frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        # Update frame counter
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
with open('chen_track_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Case_ID', 'Frame_Number', 'X', 'Y', 'Distance_Traveled', 'Speed'])
    for track_id, data in track_data.items():
        for frame_info in data:
            frame_number, x, y, distance, speed = frame_info
            writer.writerow([track_id] + list(frame_info))

