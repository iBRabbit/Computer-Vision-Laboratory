import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("img.jpg")
width, height = int(img.shape[1]), int(img.shape[0])

img = cv2.resize(img,(width, height), cv2.INTER_AREA)

# Harris Detection
harris_img = img.copy()
harris_gray_img = cv2.cvtColor(harris_img, cv2.COLOR_BGR2GRAY)

harris_img_pos = cv2.cornerHarris(harris_gray_img, blockSize=2, ksize=3, k=0.04)
harris_img_pos = cv2.dilate(harris_img_pos, None)
harris_img[harris_img_pos > 0.01 * harris_img_pos.max()] = [0, 0, 255]

_, harris_img_pos = cv2.threshold(harris_img_pos, 0.01 * harris_img_pos.max(), 255, 0)
harris_img_pos = np.uint8(harris_img_pos)
_, _, _, centroid = cv2.connectedComponentsWithStats(harris_img_pos)

subpix_img = img.copy()
subpix_img_gray = cv2.cvtColor(subpix_img, cv2.COLOR_BGR2GRAY)
corner_img = cv2.cornerSubPix(
    subpix_img_gray,
    np.float32(centroid),
    (5, 5),
    (-1, -1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
)

result = np.hstack((centroid, corner_img))
result = np.int0(result)
plt.figure(3, (20, 20))

subpix_img[result[:, 1], result[:, 0]] = [0, 255, 0]
subpix_img[result[:, 3], result[:, 2]] = [0, 0, 255]

fast = cv2.FastFeatureDetector_create()
fast_img = img.copy()
fast_img_gray = cv2.cvtColor(fast_img, cv2.COLOR_BGR2GRAY)
fast_img_kp = fast.detect(fast_img_gray)
fast_img = cv2.drawKeypoints(fast_img, fast_img_kp, None, (0,255,0))

orb = cv2.ORB_create()
orb_img = img.copy()
orb_img_gray = cv2.cvtColor(orb_img, cv2.COLOR_BGR2GRAY)
orb_img_kp = orb.detect(orb_img_gray)
orb_img = cv2.drawKeypoints(orb_img, orb_img_kp, None, (0,0,255))

labels = ["original", "harris", "subpix", "fast", "org"]
images = [img, harris_img, subpix_img, fast_img, orb_img]

plt.figure(1, (12, 8))
for i, (label, image) in enumerate(zip(labels, images)):
    col = len(images) / 2 if len(images) % 2 == 0 else len(images) / 2 + 1
    plt.subplot(2, int(col), i+1)
    plt.title(label)
    result_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(result_img)
    plt.xticks([])
    plt.yticks([])
plt.show()