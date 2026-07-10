import cv2
import pathlib
import time
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np


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

# Foreign objects code identifies black spots inside the requested object
def foreignobjects(frame):
    # 1.1 Make copy of the original image
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    original_image = frame.copy()

    # 1.2 perform Gaussian blur for smoothing
    # Bigger kernel element due to image resolution
    gaussian_grey = cv2.GaussianBlur(grey, (5, 5), 0)

    # 1.2 Perform Canny Edge Detection to locate discontinuities with hysterisis
    edged = cv2.Canny(gaussian_grey, 17, 60, L2gradient=True)

    # 1.3 Perform Closing to connect small gaps
    # Big kernel element due to image resolution
    kernel = np.ones((13, 13), np.uint8)
    closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 1.4 Take The external contours as a first step for the mask
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    max_area = cv2.contourArea(cnt)

    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)

    # 1.5 Take the 1st mask
    mask = np.zeros_like(grey, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    masked = cv2.bitwise_and(grey, grey, mask=mask)

    # 1.6 Blur the 1st mask and threshold to remove the roller conveyor
    # and dark elements and to take inner bottle
    blur = cv2.GaussianBlur(masked, (11, 11), 0)
    ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    # 1.7 Run connected components analysis
    # to take only the botlle content
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh1)

    print(stats, '\n\n', centroids)  ### !!!

    label_hue = np.uint8(179 * labels / np.max(labels))

    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # 1.8 Take the two largest components
    sizes = stats[:, -1]

    sorted_index_array = np.argsort(sizes)
    sorted_array = sizes[sorted_index_array]
    sorted_array = np.delete(sorted_array, sorted_array.argmax())

    # we want 2 largest values
    n = 2

    # find n largest value
    rslt = sorted_array[-n:]

    final = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(rslt) > 1:
        # Take the largest 2 components
        two_largest_components = np.zeros_like(grey)
        two_largest_components[labels == np.where(stats[:, -1] == rslt[0])] = 255
        two_largest_components[labels == np.where(stats[:, -1] == rslt[1])] = 255

        # 1.9 After the final mask to take the content
        final_masked = cv2.bitwise_and(grey, grey, mask=two_largest_components)

        # 1.10 Take the content edges
        content_edged = cv2.Canny(final_masked, 17, 60)

        contours, hierarchy = cv2.findContours(content_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        drawing_img = np.zeros_like(content_edged)
        if len(contours) > 0:
            cv2.drawContours(drawing_img, contours, 0, (255), 1)
        else:
            drawing_img = frame

        # 1.11 Take the masked content and remove light reflection
        final_content = cv2.bitwise_and(grey, grey, mask=content_edged)

        # 1.12 Thresold to remove the whites
        ret, thresh7 = cv2.threshold(final_content, 156, 255, cv2.THRESH_BINARY)

        # 1.13 Remove whites from the edged
        for i in range(content_edged.shape[0]):
            for j in range(content_edged.shape[1]):
                if thresh7[i][j] == 255:
                    content_edged[i][j] = 0

        # 1.14 Show the foreign objects
        final_product = cv2.bitwise_and(grey, grey, mask=content_edged)

        # 1.15 Absdiff to show the foreign objects to the final pic
        final = cv2.absdiff(grey, final_product)

    return final

# Liquid level function checks if the liquid inside the bottle is inside the necessary levels
def liquidlevel(roi):
    original_frame = roi
    roi = cv2.cvtColor(roi, 0)

    # crop image just for the bottle neck
    cropped_image = roi[192:250, 0:300]

    # Take the canny edges
    edges = cv2.Canny(cropped_image, 252, 119)

    # Draw the hough lines
    # Hough lines parameters
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(cropped_image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    frame_liquid_level = 0

    # Draw the lines
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw the lines on the  image
        lines_edges = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)

        plt.imshow(lines_edges)
        plt.show()

        # Take the lowest line
        y_avgs = [(line[0, 1] + line[0, 3]) / 2 for line in lines]

        # the x-axis valuse which it the liquid level
        if y_avgs is not None:
            frame_liquid_level = max(y_avgs)

    return frame_liquid_level

# Liquid level check for video visualization
def liquid_level_check(liquid, sketcher):
    # check for existance of liquid
    if liquid != 192:
        cv2.line(sketcher, pt1=(0, int(liquid)), pt2=(300, int(liquid)), color=(0, 0, 255),
                 thickness=1)
        cv2.line(sketcher, pt1=(0, 225), pt2=(300, 225), color=(255, 0, 0), thickness=1)
        cv2.line(sketcher, pt1=(0, 247), pt2=(300, 247), color=(255, 0, 0), thickness=1)
        # minimun and maximum border for liquid level check
        if liquid > 225 and liquid < 247:
            cv2.putText(img=sketcher, text='Liquid Level : OK ', org=(10, 20),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)
        else:
            cv2.putText(img=sketcher, text='Liquid Level : NOT OK ', org=(10, 20),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1)

# Construct the argument parser.
parser = argparse.ArgumentParser()

# Import the video to be tested
parser.add_argument(
    '-i', '--input', help='path to input video',
    default='20221014_155146-trim.mp4'
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

# Set the count frame
frame_count = 0  # To count total frames.
total_fps = 0  # To get the final frames per second.

# Static roi
upper_left = (910, 300)
bottom_right = (1110, 1000)

# Static roi0
upper_left0 = (600, 300)
bottom_right0 = (800, 1000)

# Read until end of video.
while (cap.isOpened()):

    # Capture each frame of the video.
    ret, frame = cap.read()

    if ret:
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # count start for fps visualization
        start_time = time.time()

        # Set our first region of interest
        roi0 = frame[300:1000, 600: 800]
        roi = frame[300:1000, 910: 1110]

        # Replacing the sketched image on first Region of Interest
        rect_img0 = frame[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]]

        sketcher_rect0 = rect_img0

        # Use sketch transform function to convert again to grayscale
        sketcher_rect0 = sketch_transform(sketcher_rect0)

        # Take the area of interest and check for foreign objects
        sketcher_rect_rgb0 = foreignobjects(roi0)

        # Conversion for 3 channels to put back on original image (streaming)
        sketcher_rect_rgb0 = cv2.cvtColor(sketcher_rect_rgb0, cv2.COLOR_GRAY2RGB)

        frame[upper_left0[1]: bottom_right0[1], upper_left0[0]: bottom_right0[0]] = sketcher_rect_rgb0

        # Replacing the sketched image on  second Region of Interest
        rect_img = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

        sketcher_rect = rect_img

        sketcher_rect = sketch_transform(sketcher_rect)

        # Conversion for 3 channels to put back on original image (streaming)
        sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)

        # In the second area of interest check the liquid level
        liquid_lvl = liquidlevel(roi) + 192

        liquid_level_check(liquid_lvl, sketcher_rect_rgb)

        # Replacing the sketched image on Region of Interest
        frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = sketcher_rect_rgb

        end_time = time.time()

        # Get the current fps.
        fps = 1 / (end_time - start_time)

        # Add `fps` to `total_fps`.
        total_fps += fps

        # Increment frame count.
        frame_count += 1

        # Black spot detection and warning message
        roi2 = frame[570:850, 600:780]
        count = 0
        for i in range(roi2.shape[0]):
            for j in range(roi2.shape[1]):
                if roi2[i][j].all() == 0:
                    count += 1
        # If black spots are more that the maximum number of allows black pixels (extracted from a clear bottle), then
        # error appears.
        print(count)
        if count > 900:
            print('DIFECTIVE PRODUCT')
            cv2.putText(img=frame, text='Warning: DEFECTIVE PRODUCT', org=(100, 100),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255), thickness=2)

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

# Close all frames and video windows.
cv2.destroyAllWindows()

