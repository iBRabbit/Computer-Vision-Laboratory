import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.jpg')
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = image.shape[:2]

def showResult (n_row = 0, n_col = 0, res_stack = None) :
    plt.figure(figsize = (12, 12))
    for idx, (lbl, img) in enumerate(res_stack):
        plt.subplot(n_row, n_col, idx + 1)
        plt.imshow(img, 'gray')   
        plt.title(lbl) 
        plt.axis('off')
    plt.show()
            
# Laplacian
laplace_8 = cv2.Laplacian(imgray, cv2.CV_8U)
laplace_16 = cv2.Laplacian(imgray, cv2.CV_16S)
laplace_32 = cv2.Laplacian(imgray, cv2.CV_32F)
laplace_64 = cv2.Laplacian(imgray, cv2.CV_64F)

laplace_labels = ['Laplace 8 Bit', 'Laplace 16 Bit', 'Laplace 32 Bit', 'Laplace 64 Bit']
laplace_images = [laplace_8, laplace_16, laplace_32, laplace_64]

# showResult(2, 2, zip(laplace_labels, laplace_images))

# Kernel : Matriks yang dipake buat ngitung

# Sobel
# Gx = [] * kernel X
# Gy = [] * kernel Y
# Biar perfecto = sqrt(gx^2 + gy^2)

def calculateSobel(src, kernel, ksize) :
    res_matrix = np.array(src)
    for i in range(height - ksize - 1) : 
        for j in range(width - ksize - 1) :
            patch = src[i : (i + ksize), j : (j + ksize)].flatten()
            result = np.convolve(patch, kernel, 'valid')
            res_matrix[i + ksize // 2, j + ksize // 2] = result[0]    
    return res_matrix
    
kernelX = np.array([
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
])

kernelY = np.array([
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
])

ksize = 3
manual_sobel_x = imgray.copy()
manual_sobel_y = imgray.copy()

manual_sobel_x = calculateSobel(manual_sobel_x, kernelX, ksize)
manual_sobel_y = calculateSobel(manual_sobel_y, kernelY, ksize)

sobel_x = cv2.Sobel(imgray, cv2.CV_32F, 1, 0, ksize = 3)
sobel_y = cv2.Sobel(imgray, cv2.CV_32F, 0, 1, ksize = 3)

sobel_labels = ['manual sobel x', 'manual sobel y', 'sobel x', 'sobel y']
sobel_images = [manual_sobel_x, manual_sobel_y, sobel_x, sobel_y]

# showResult(2, 2, zip(sobel_labels, sobel_images))

merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
merged_sobel *= 255.0 / merged_sobel.max()

merged_manual_sobel = cv2.bitwise_or(manual_sobel_x, manual_sobel_y)
merged_manual_sobel = np.uint16(np.abs(merged_manual_sobel))

merged_sobel_labels = ['merged sobel', 'manual merged sobel']
merged_sobel_images = [merged_sobel, merged_manual_sobel]

showResult(1, 2, zip(merged_sobel_labels, merged_sobel_images))

# Canny
# Hasil output sobel => Input Canny

# Threshold biasanya 2 :1 atau 3 : 1
canny_50_100 = cv2.Canny(imgray, 50, 100)
canny_50_150 = cv2.Canny(imgray, 50, 150)
canny_75_150 = cv2.Canny(imgray, 75, 150)
canny_75_225 = cv2.Canny(imgray, 75, 225)

canny_labels = ['canny 50 100', 'canny 50 150', 'canny 75 150', 'canny 75 225']
canny_images = [canny_50_100, canny_50_150, canny_75_150, canny_75_225]

showResult(2,2, zip(canny_labels, canny_images))
