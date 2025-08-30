import cv2
import numpy as np

# Define the color ranges in HSV for yellow, blue, and red
# Lower and upper bounds for Red, Blue, and Yellow in HSV color space
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Function to detect colors
def detect_colors(frame):
    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Combine both red masks
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply the masks to the original image to isolate the colors
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)
    result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    result_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    return result_red, result_blue, result_yellow, mask_red, mask_blue, mask_yellow

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Detect the colors in the frame
    result_red, result_blue, result_yellow, mask_red, mask_blue, mask_yellow = detect_colors(frame)

    # Display the original frame, masks, and color detection results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Red Mask', mask_red)
    cv2.imshow('Blue Mask', mask_blue)
    cv2.imshow('Yellow Mask', mask_yellow)
    cv2.imshow('Red Detected', result_red)
    cv2.imshow('Blue Detected', result_blue)
    cv2.imshow('Yellow Detected', result_yellow)

    # Break the loop if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
