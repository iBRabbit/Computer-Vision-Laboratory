import cv2
import matplotlib.pyplot as plt

def show_image(name, img) :
    cv2.imshow(name, img)
    cv2.waitKey(0)

# Gray Scaling

# Cara 1 - Grayscaling
# image = cv2.imread("cat.jpg", 0)

# Cara 2 - Grayscaling
image = cv2.imread("cat.jpg")

width = int(image.shape[1] * 0.50) 
height = int(image.shape[0] * 0.50)
image = cv2.resize(image, (width, height))

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show_image("Kucing", gray_image)

# Blur
blur_img = cv2.blur(image, (5, 5))
# show_image("Blur", blur_img)

gaussian_img = cv2.GaussianBlur(image, (15, 15), 0)
# show_image("Gaussian Blur", gaussian_img)

median_img = cv2.medianBlur(gray_image, 5);
# show_image("Median Blur", median_img)

bilaterial_imsg = cv2.bilateralFilter(gray_image, 10, 10, 10)
# show_image("Bilateral Filter",bilaterial_imsg)

# threshold
# Ngubah pixel src(x,y) > thresh pilih maxval else 0
_, binary_img = cv2.threshold(gray_image, 127, 200, cv2.THRESH_BINARY)
# show_image("Binary Image", binary_img)

_, binary_inv = cv2.threshold(gray_image, 127, 200, cv2.THRESH_BINARY_INV)
# show_image("Binary Inverse", binary_inv)

_, binary_tozero = cv2.threshold(gray_image, 127, 200, cv2.THRESH_TOZERO)
# show_image("Binary Inverse", binary_tozero)

_, binary_tozero_inv = cv2.threshold(gray_image, 127, 200, cv2.THRESH_TOZERO_INV)
# show_image("binary_tozero_inv", binary_tozero_inv)

_, truncated_img = cv2.threshold(gray_image, 255, 255, cv2.THRESH_TRUNC)
# show_image("truncated_img", truncated_img)

_, triangle_img = cv2.threshold(gray_image, 50, 255, cv2.THRESH_TRIANGLE)
# show_image("TRIANGLE", triangle_img)

_, otsu_img = cv2.threshold(gray_image, 255, 255, cv2.THRESH_OTSU)
# show_image("otsu", otsu_img)  

threshold_img = [
        binary_img, binary_inv, binary_tozero, 
        binary_tozero_inv, truncated_img, triangle_img, otsu_img
    ]

threshold_label = [
    "Binary",
    "Binary Invert",
    "Binary to Zero",
    "Truncated",
    "TRIANGLE",
    "Otsu"
]

plt.figure(1, (10, 5))

for i, (label, img) in enumerate(zip(threshold_label, threshold_img)) : 
    plt.subplot(4, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(label)
    plt.axis('off')
    
plt.show()
    