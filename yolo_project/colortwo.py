import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
yolo = YOLO('yolov8s.pt')

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Function to detect color
def detect_color(frame, x1, y1, x2, y2):
    # Crop the detected object
    cropped = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
    "red": ([136, 87, 111], [180, 255, 255]),
    "green": ([40, 70, 70], [80, 255, 255]),      # Refined green
    "blue": ([90, 50, 70], [130, 255, 255]),      # Deep blue
    "light_blue": ([81, 50, 70], [99, 255, 255])  # Light blue / cyan
}


    

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:
            return color_name
    return "unknown"

# Load video capture
videoCap = cv2.VideoCapture(0)

while True:
    ret, frame = videoCap.read()
    if not ret:
        continue

    results = yolo.track(frame, stream=True)

    for result in results:
        classes_names = result.names

        for box in result.boxes:
            if box.conf[0] > 0.4:
                [x1, y1, x2, y2] = box.xyxy[0].int().tolist()
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                colour = getColours(cls)

                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Detect color
                detected_color = detect_color(frame, x1, y1, x2, y2)

                # Put class name and detected color on the image
                cv2.putText(frame, f'{class_name} {detected_color}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # Show the image
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()