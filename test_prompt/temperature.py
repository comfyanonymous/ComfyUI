import cv2
import numpy as np
import requests
import time

# Function to draw the fixed parts of the thermometer (outline and labels)
def draw_fixed_parts(img):
    # Draw the outline of the thermometer
    outline = np.array([[250, 50], [275, 50], [275, 460], [225, 460], [225, 50], [250, 50]], np.int32)
    cv2.polylines(img, [outline], isClosed=True, color=(0, 0, 0), thickness=5)

    # Add temperature labels
    cv2.putText(img, '0', (180, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, '40', (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Draw the bulb
    cv2.circle(img, (250, 460), 50, (0, 0, 0), 5)

# Function to update the filled bulb color
def fill_bulb(img, color):
    cv2.circle(img, (250, 460), 50, color, -1)

# Function to update the temperature bar and color
def update_temperature_bar(img, temp, color):
    top = int(460 - 410 * temp / 40)
    bar = np.array([[250, 460], [270, 460], [270, top], [230, top], [230, 460], [250, 460]], np.int32)
    cv2.fillPoly(img, [bar], color)
    cv2.putText(img, f'{temp:.1f}', (330, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

# Function to create the thermometer image
def draw_thermometer(temperature):
    img = np.ones((600, 600, 3), dtype=np.uint8) * 255

    draw_fixed_parts(img)

    # color_map = [(1, (255, 191, 0)), (11, (173, 255, 47)), (21, (255, 165, 0)), (31, (255, 0, 0)), (40, (255, 0, 0))]
    color_map = [(11, (255, 255, 0)), (21, (144, 238, 144)), (31, (0, 127, 255)), (40, (0, 0, 255))]
    color = (0, 0, 255)
    for start_temp, start_color in color_map:
        if temperature <= start_temp:
            color = start_color
            break

    update_temperature_bar(img, temperature, color)
    fill_bulb(img, color)

    return img

# Function to overlay images with transparency
def overlay_images(background, overlay, x_offset, y_offset, alpha=0.5):
    y1, y2 = y_offset, y_offset + overlay.shape[0]
    x1, x2 = x_offset, x_offset + overlay.shape[1]

    # Ensure the overlay fits within the background
    if y2 > background.shape[0]:
        y2 = background.shape[0]
        overlay = overlay[:y2 - y1, :]
    if x2 > background.shape[1]:
        x2 = background.shape[1]
        overlay = overlay[:, :x2 - x1]

    # Blend images
    background[y1:y2, x1:x2] = cv2.addWeighted(background[y1:y2, x1:x2], 1 - alpha, overlay, alpha, 0)
    return background
