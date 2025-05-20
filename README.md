# Real-Time Crowd Detection using Live Webcam Feed

## Project Overview
This project implements a **real-time crowd detection system** using your laptop's webcam. The system detects persons in the live video feed, analyzes their proximity, and identifies crowds based on the logic:
- A crowd is defined as **three or more persons standing close together** for **10 consecutive frames**.
- When such a crowd is detected, the system logs the event with the frame number and the number of persons detected.

This solution leverages a pre-trained object detection model (YOLOv5) for detecting people and custom logic for crowd detection and persistence analysis.

---

## Features
- **Live detection** from webcam video feed (no need for pre-recorded videos).
- Detects persons using a state-of-the-art YOLOv5 object detection model.
- Identifies groups of 3 or more people standing close together.
- Tracks groups over 10 consecutive frames to confirm crowd formation.
- Logs detected crowd events (frame number and crowd size) into a CSV file.
- Real-time visual feedback with bounding boxes and crowd alerts.

---

## How It Works

1. **Capture live video** from the webcam using OpenCV.
2. **Run YOLOv5 model** on each frame to detect persons.
3. **Calculate distances** between detected persons to find groups standing close.
4. **Track group persistence** across consecutive frames.
5. If a group of 3+ persists for 10 frames, **log the crowd event**.
6. Display live results on the screen with bounding boxes and counts.

---

## Technologies Used

- Python 3.x
- OpenCV (for webcam feed and image processing)
- PyTorch & YOLOv5 (for person detection)
- NumPy (for distance calculations)
- Pandas or CSV (for logging events)

---

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crowd-detection-live.git
   cd crowd-detection-live
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the detection script:

bash
Copy
Edit
python crowd_detection_live.py
The program will open your webcam and start detecting crowds in real-time.
Detected crowd events will be logged in crowd_log.csv.

How to Use
Ensure your laptop camera is functional.

Run the script and allow camera access.

Move around or simulate groups standing close to test crowd detection.

Watch the live video with bounding boxes and crowd alerts.

Check the crowd_log.csv file for logged crowd events.

Future Work
Add alert notifications (e.g., sound or message popups).

Optimize for higher frame rates and lower latency.

Extend to detect crowd density and movement patterns.

Deploy on edge devices or integrate with CCTV systems.

