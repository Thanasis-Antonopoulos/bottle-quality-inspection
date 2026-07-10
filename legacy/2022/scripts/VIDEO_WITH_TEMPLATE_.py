import cv2
import pathlib
import time
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np

# code for multiple template matching https://pyimagesearch.com/2021/03/29/multi-template-matching-with-opencv/
# https://www.geeksforgeeks.org/multi-template-matching-with-opencv/
# https://stackoverflow.com/questions/56667067/opencv-template-matching-multiple-templates

# With the function bellow we can find the precise template that we want. We will not implement this as we will
# follow the default threshold of the template match which is around 50%. As in our case the video conditions are not
# optimal. This function is extremely useful if we want to find multiple templates of different objects with high
# precision

def matching_template(roi):
    original_frame = roi
    gray_image = cv2.cvtColor(roi, 0)

    # image = cv2.imread('felos.jpg')

    template_corck = cv2.imread('correct_cork.jpg')

    # orig_image = image.copy()
    orig_template = template_corck.copy()

    # gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(orig_template, cv2.COLOR_BGR2GRAY)

    # perform template matching
    print("[INFO] performing template matching...")
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    print("Result matrix", "\n", result, "\n",
          "\n",
          "Minimum value of matrix", "\n", minVal, "\n",
          "\n",
          "Maximum value of matrix", "\n", maxVal, "\n",
          "\n",
          "Minimum location of template match", "\n", minLoc, "\n",
          "\n",
          "Maximum location of template match", "\n", maxLoc)

    w, h = gray_template.shape[::-1]

    res = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        try:
            # draws
            found = cv2.rectangle(gray_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
            print("Template match found with success of " + threshold)
            return found
        except Exception as e:
            print("Template not found. Problem on cork", '\n' + repr(e))


# Sketch format creates the grayscale area in our region of interest. With this function we can also implement other
# transformations like blurring, thresholding etc.

def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    # image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    # _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return image_grayscale


# Construct the argument parser.
parser = argparse.ArgumentParser()

# Import the video to be tested
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='20221014_155146-trim.mp4'
)

# Import the first template for the cork
parser.add_argument(
    '-t', '--template', help='path to the template',
    default='correct_cork.jpg'
)

# Import the second template for the bottle
parser.add_argument(
    '-t1', '--template1', help='path to the template',
    default='bottle_template.jpg'
)
args = vars(parser.parse_args())

# Read the video input.
cap = cv2.VideoCapture(args['input'])

# If there is a problem with the video input
if OSError:
    print('Program has stopped')

# Get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# String name with which to save the resulting video.
save_name = str(pathlib.Path(
    args['input']
)).split(os.path.sep)[-1].split('.')[0]

# define codec and create VideoWriter object
out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

# Read the template in grayscale format.
template = cv2.imread(args['template'], 0)
w, h = template.shape[::-1]

# Read the template1 in grayscale format.
template1 = cv2.imread(args['template1'], 0)
w1, h1 = template1.shape[::-1]

# Set the count frame
frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

# Read until end of video.
while (cap.isOpened()):

    # Capture each frame of the video.
    ret, frame = cap.read()

    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start_time = time.time()

        # Apply template Matching.
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        result1 = cv2.matchTemplate(image, template1, cv2.TM_CCOEFF_NORMED)

        # Template for cork
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Top left x and y coordinates.
        x1, y1 = max_loc

        # Bottom right x and y coordinates.
        x2, y2 = (x1 + w, y1 + h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Template for bottle
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1)

        # Top left x and y coordinates.
        x1_1, y1_1 = max_loc1

        # Bottom right x and y coordinates.
        x2_1, y2_1 = (x1_1 + w1, y1_1 + h1)
        cv2.rectangle(frame, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 2)

        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)

        # Add `fps` to `total_fps`.
        total_fps += fps

        # Increment frame count.
        frame_count += 1

        # Frame counter visualization
        cv2.putText(frame, f"{fps:.1f} FPS",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, lineType=cv2.LINE_AA)
        cv2.imshow('Result', frame)
        out.write(frame)

        # Press esc to exit.
        key = cv2.waitKey(30)
        if key == 27:
            break
    else:
        break

# Release VideoCapture() object.
cap.release()

out.release()

# Close all frames and video windows.
cv2.destroyAllWindows()
