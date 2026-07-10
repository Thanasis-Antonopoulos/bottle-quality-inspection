import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


# import of original image and template
image = cv2.imread('felos.jpg')
template_corck = cv2.imread('correct_cork.jpg')

# copy to be able to preprocess and keep the original
orig_image = image.copy()
orig_template = template_corck.copy()

# grayscale to be able to perform template match
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(orig_template, cv2.COLOR_BGR2GRAY)

# #visualising
titles = ['Grayed Image', 'Grayed Image template']
images = [gray_image, gray_template]

for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# perform template matching
print("[INFO] performing template matching...")
result = cv2.matchTemplate(gray_image, gray_template,cv2.TM_CCOEFF_NORMED)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

# Printing of the template results. Min value is the negative template match, max value is the best template match
# perecentage. Minimum and maximum location is where the worst/ best match is found.

print("Result matrix","\n",result,"\n",
      "\n",
      "Minimum value of matrix","\n",minVal,"\n",
      "\n",
      "Maximum value of matrix","\n",maxVal,"\n",
      "\n",
      "Minimum location of template match","\n",minLoc,"\n",
      "\n",
      "Maximum location of template match","\n",maxLoc)


w, h = gray_template.shape[::-1]
# Perform second template match for visualization reasons. With the threshold variable we can set the desirable
# accuracy, if it not exceeds the max value
res = cv2.matchTemplate(gray_image,gray_template,cv2.TM_CCOEFF_NORMED)
threshold = 0.80
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    try:
        # draws
        found = cv2.rectangle(gray_image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    except Exception as e:
        print("Template not found. Problem on cork",'\n'+ repr(e))

# #visualising
plt.imshow(found)
plt.title('Found template with accuracy of 80%')
plt.xticks([]),plt.yticks([])
plt.show()

titles = ['Template', 'Found template']
images = [gray_template, found]

for i in range(2):
    plt.subplot(1,2,i+1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



