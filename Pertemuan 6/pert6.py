import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read
img_obj = cv.imread('box.png')
img_scn = cv.imread('box_in_scene.png')

# Bikin object untuk detector, SIFT, AKAZE, ORB
SIFT = cv.SIFT_create()
AKAZE = cv.AKAZE_create()
ORB = cv.ORB_create()

# Object mendeteksi keypoint & descriptor
# Parameter 1 : src image
# Parameter 2 : masking 
sift_kp_obj, sift_ds_obj = SIFT.detectAndCompute(img_obj, None)
sift_kp_scn, sift_ds_scn = SIFT.detectAndCompute(img_scn, None)

akaze_kp_obj, akaze_ds_obj = AKAZE.detectAndCompute(img_obj, None)
akaze_kp_scn, akaze_ds_scn = AKAZE.detectAndCompute(img_scn, None)

orb_kp_obj, orb_ds_obj = ORB.detectAndCompute(img_obj, None)
orb_kp_scn, orb_ds_scn = ORB.detectAndCompute(img_scn, None)

print(sift_kp_obj[0].pt) # Return Koordinat

# Butuh nearest neighbour
# Library -> ubah dulu jadi Float
sift_ds_obj = np.float32(sift_ds_obj)
sift_ds_scn = np.float32(sift_ds_scn)
akaze_ds_obj = np.float32(akaze_ds_obj)
akaze_ds_scn = np.float32(akaze_ds_scn)

# Brute Force untu membandingkan obj sama scn
# Algorithm = 1, check = 50
# Euclidian
flann = cv.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))

# ORB
# 1 Titik yang udah pernah di traverse, 2 blom di traverse akan tetap di kalkulasi
# Hamming Distance

bf_matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)

# Object di scene matcher
# Cocokin descriptor di object sama yg di scene
# cari 2 yang match terbaik
sift_match = flann.knnMatch(sift_ds_obj, sift_ds_scn, 2)
akaze_match = flann.knnMatch(akaze_ds_obj, akaze_ds_scn, 2)
orb_match = bf_matcher.match(orb_ds_obj, orb_ds_scn)
# Orb ; Sorted
orb_match = sorted(orb_match, key = lambda x : x.distance)

# knn match => bisa aja ada yg bener2 sesuai dan bisa aja ada yang salah
sift_matches_mask = [[0, 0] for i in range(0, len(sift_match))]
akaze_matches_mask = [[0, 0] for i in range(0, len(akaze_match))]
# print(sift_matches_mask)

def createMask(mask, match) :
    for i, (fm, sm) in enumerate(match) :
        if fm.distance < 0.7 * sm.distance :
            mask[i] = [1,0]
            
    return mask

sift_matches_mask = createMask(sift_matches_mask, sift_match)
akaze_matches_mask = createMask(akaze_matches_mask, akaze_match)

print(sift_matches_mask)

sift_res = cv.drawMatchesKnn(
    img_obj, sift_kp_obj,
    img_scn, sift_kp_scn,
    sift_match, None,
    matchColor = [255, 0, 0],
    singlePointColor = [0, 255, 0],
    matchesMask = sift_matches_mask
)

akaze_res = cv.drawMatchesKnn(
    img_obj, akaze_kp_obj,
    img_scn, akaze_kp_scn,
    akaze_match, None,
    matchColor = [255, 0, 0],
    singlePointColor = [0, 255, 0],
    matchesMask = akaze_matches_mask
)

orb_res = cv.drawMatches(
    img_obj, orb_kp_obj,
    img_scn, orb_kp_scn,
    
    # 20 index pertama
    orb_match[:20], None,
    matchColor = [255, 0, 0],
    singlePointColor = [0, 255, 0],
    flags = 2
)

matching_labels = ['sift', 'akaze', 'orb']
matching_images = [sift_res, akaze_res, orb_res]


plt.figure(figsize=(12,12))
for i, (lbl, image) in enumerate(zip(matching_labels, matching_images)) :
    plt.subplot(2, 2, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(lbl)
    
plt.show()