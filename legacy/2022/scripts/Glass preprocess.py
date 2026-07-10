import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Import, copy and grayscale original image
image = cv2.imread('broken_glass.jpg')

orig_image = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# visualising
titles = ['Original Image', 'Grayed Image']
images = [orig_image, gray]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Initialize a list of kernel sizes (so we can evaluate the relationship between kernel
# size and amount of blurring)

# kernelSizes = [(3, 3), (9, 9), (15, 15)]
# # loop over the kernel sizes
# for (kX, kY) in kernelSizes:
# 	# apply an "average" blur to the image using the current kernel
# 	# size
# 	blurred = cv2.blur(gray, (kX, kY))
# 	cv2.imshow("Average ({}, {})".format(kX, kY), blurred)
# 	cv2.waitKey(0)

# perform median blur to remove noise
blurred = cv2.blur(gray, (3, 3))

bedged = cv2.Canny(blurred, 17, 60, L2gradient=True)
bwindow_name = str(17) + '^' + str(60)

# visualising
titles = ['Grayed image blurred', 'Grayed image blurred + canny']
images = [blurred, bedged]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# plt.imshow(blurred)
# plt.title('Grayed image blurred')
# plt.xticks([]), plt.yticks([])
# plt.show()

# histogram of blurred image to find pixel values
plt.hist(blurred.ravel(), 256, [0, 256]);
plt.title("Grayed image blurred Histogram")
plt.show()

# plt.imshow(bedged)
# plt.title('Grayed image blurred + canny')
# plt.xticks([]), plt.yticks([])
# plt.show()

# plt.hist(bedged.ravel(), 256, [0, 256]);
# plt.title("Grayed image blurred + canny histogram")
# plt.show()

# # close all windows to cleanup the screen
# cv2.destroyAllWindows()
# cv2.imshow("Original", image)
# loop over the kernel sizes again
# for (kX, kY) in kernelSizes:
# 	# apply a "Gaussian" blur to the image
# 	blurred = cv2.GaussianBlur(gray, (kX, kY), 0)
# 	cv2.imshow("Gaussian ({}, {})".format(kX, kY), blurred)
# 	cv2.waitKey(0)

# applying Gaussian Blur for bette results
GaussianBlur = cv2.GaussianBlur(gray, (3, 3), 0)

edged = cv2.Canny(GaussianBlur, 17, 60, L2gradient=True)
window_name = str(17) + '^' + str(60)

# visualising
titles = ['Grayed image GaussianBlur', 'Grayed image GaussianBlur canny']
images = [GaussianBlur, edged]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# plt.imshow(GaussianBlur)
# plt.title('Grayed image GaussianBlur')
# plt.xticks([]), plt.yticks([])
# plt.show()

# histogram to double check any changes
plt.hist(GaussianBlur.ravel(), 256, [0, 256]);
plt.title("Grayed image GaussianBlur histogram")
plt.show()

# plt.imshow(edged)
# plt.title('Grayed image GaussianBlur canny')
# plt.xticks([]), plt.yticks([])
# plt.show()

# plt.hist(edged.ravel(), 256, [0, 256]);
# plt.title("Grayed image GaussianBlur canny histogram")
# plt.show()

# Thresholding

(T, thresh) = cv2.threshold(GaussianBlur, 100, 255, cv2.THRESH_BINARY)

(T, threshInv) = cv2.threshold(GaussianBlur, 100, 255, cv2.THRESH_BINARY_INV)

# visualising
titles = ['THRESH_BINARY', 'threshInv']
images = [thresh, threshInv]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()


#--------------------------------------------------
plt.imshow(thresh)
plt.title('Binary Threshold')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(threshInv)
plt.title('Binary Threshold Inverted')
plt.xticks([]), plt.yticks([])
plt.show()

plt.hist(thresh.ravel(), 256, [0, 256]);
plt.title("THRESH_BINARY histogram")
plt.show()

plt.hist(threshInv.ravel(), 256, [0, 256]);
plt.title("threshInv histogram")
plt.show()

#---------------------------------------------------------

# apply Otsu's automatic thresholding which automatically determines
# the best threshold value
(T, Otsu) = cv2.threshold(GaussianBlur, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

print("[INFO] otsu's thresholding value: {}".format(T))
# visualize only the masked regions in the image
masked = cv2.bitwise_and(image, image, mask=Otsu)

# visualising
titles = ['Otsu', 'Otsu masked']
images = [Otsu, masked]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(Otsu)
plt.title('Otsu with Gaussina blur and inverted binary thresholding')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(masked)
plt.title('Otsu masked')
plt.xticks([]), plt.yticks([])
plt.show()

# download original image
img = cv2.imread('broken_glass.jpg', 0)

# can blur (source, kernel size)
img = cv2.medianBlur(img, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# no ret value for adaptive
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# again for visualising
titles = ['Original Image', 'Global Binary Threshold', 'Adaptive Mean Threshold', 'Adaptive Gaussian Threshold']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

