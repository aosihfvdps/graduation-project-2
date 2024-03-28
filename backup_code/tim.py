import csv
def detect_anomalies(track_history, frame_number):
    anomalies = defaultdict(list)
    for track_id, track in track_history.items():
        # 速度異常
        if len(track) >= 2:
            speeds = [np.sqrt((track[i][0]-track[i-1][0])**2 + (track[i][1]-track[i-1][1])**2) for i in range(1, len(track))]
            avg_speed = np.mean(speeds)
            if avg_speed > MAX_SPEED_THRESHOLD or avg_speed < MIN_SPEED_THRESHOLD:
                anomalies[track_id].append(f"Speed anomaly detected at frame {frame_number}. Avg speed: {avg_speed:.2f}")
        
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