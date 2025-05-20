import cv2
import torch
import numpy as np
from scipy.spatial.distance import cdist
from collections import deque
import csv

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Constants
PERSON_CLASS_ID = 0
DISTANCE_THRESHOLD = 75
FRAMES_REQUIRED = 10
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_number = 0

# Deque to store last 10 crowd detections (True/False)
crowd_frame_history = deque(maxlen=FRAMES_REQUIRED)
crowd_logged = False  # Prevent logging multiple times for same crowd event

# CSV setup
csv_filename = "crowd_detection_log.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Frame Number', 'Person Count in Crowd'])  # Header

# Helper: get center points of bounding boxes
def get_centers(boxes):
    return np.array([[(x1 + x2) / 2, (y1 + y2) / 2] for x1, y1, x2, y2 in boxes])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1

    # YOLOv5 inference
    results = model(frame)

    # Extract person detections
    persons = []
    for *box, conf, cls in results.xyxy[0].tolist():
        if int(cls) == PERSON_CLASS_ID:
            persons.append(box)  # [x1, y1, x2, y2]

    centers = get_centers(persons)
    groups = []

    # Group people based on proximity
    if len(centers) >= 3:
        distances = cdist(centers, centers)
        for i in range(len(centers)):
            close = np.where(distances[i] < DISTANCE_THRESHOLD)[0]
            if len(close) >= 3:
                group = tuple(sorted(close))
                if group not in groups:
                    groups.append(group)

    # Determine if a crowd is present in this frame
    crowd_detected = len(groups) > 0
    crowd_frame_history.append(crowd_detected)

    # If crowd present for 10 consecutive frames and not logged yet
    if sum(crowd_frame_history) == FRAMES_REQUIRED and not crowd_logged:
        crowd_logged = True
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([frame_number, len(centers)])
        print(f"CROWD DETECTED at frame {frame_number} with {len(centers)} persons")

    # Reset logging flag if no crowd
    if sum(crowd_frame_history) < FRAMES_REQUIRED:
        crowd_logged = False

    # Draw detections
    for box in persons:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if sum(crowd_frame_history) == FRAMES_REQUIRED:
        cv2.putText(frame, "CROWD DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)

    cv2.imshow('Crowd Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
