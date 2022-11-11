import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Test", img_gray)
cv2.waitKey(0)

width = img.shape[1]
height = img.shape[0]

intensity = np.zeros(256, dtype=int)

for i in range(height) :
    for j in range(width):
        intensity[img_gray[i,j]] += 1
        
plt.plot(intensity, 'g o', label = 'Japan View')
plt.xlabel("Intensity")
plt.ylabel("Quantity")
plt.legend(loc="upper left")
plt.show()

equalized_img = cv2.equalizeHist(img_gray)
equalized_intensity = np.zeros(256, dtype=int)

for i in range(height) :
    for j in range(width):
        equalized_intensity[equalized_img[i,j]] += 1
        
plt.figure(1, (16, 8))
plt.subplot(1, 2, 1)
plt.plot(intensity, 'g', label = 'Japan View')
plt.xlabel("Intensity")
plt.ylabel("Quantity")
plt.legend(loc="upper left")


plt.subplot(1, 2, 2)
plt.plot(equalized_intensity, 'g', label = 'Japan View')
plt.xlabel("Intensity")
plt.ylabel("Quantity")
plt.legend(loc="upper left")
plt.show()

# Clahe Equalization
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img_gray)

res = np.vstack((img_gray, equalized_img, clahe_img))
cv2.imshow("Equalization", res)
cv2.waitKey(0)