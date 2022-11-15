import cv2 
import matplotlib.pyplot as plt
import numpy as np

def show_image(img, title = "Default", gray = True) :
    plt.figure(1, (16, 8))
    plt.title(title)
    if gray : plt.imshow(img, cmap='gray')
    else : 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_stacked_images(arr_imgs, arr_labels, stack_row = 4, stack_col = 3, gray = True) :
    plt.figure(1, (16, 8))
    for i, (label, image) in enumerate(zip(arr_labels, arr_imgs)) :
        plt.subplot(stack_row, stack_col, (i + 1))
        plt.title(label)
        
        if gray : 
            plt.imshow(image, cmap='gray')
        else : 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)

        plt.axis('off')
    plt.show()

def read_all_images() :
    
    images = []
    labels = []
    
    for i in range(1, 11) :
        file_name = f"sample{i}.jpg"    
        path = 'Dataset/Data/' + file_name
        tmp = cv2.imread(path)
        images.append(tmp)
        labels.append(file_name)
        print(f"File {path} successfully read.")
    
    return images, labels

def read_sample_image(i) :
    path = f"Dataset/Data/sample{i}.jpg"
    img = cv2.imread(path)
    return img

def read_target() :
    path = f"Dataset/Target.jpg"
    img = cv2.imread(path)
    return img

def equalize_image(img) :
    clone_img = img.copy()
    gray_img = cv2.cvtColor(clone_img, cv2.COLOR_RGB2GRAY)
    equalized_img = cv2.equalizeHist(gray_img)
    
    return equalized_img

def equalize_all_images(arr_img) :
    images = []
    for i in arr_img :
        new_img = equalize_image(i)
        images.append(new_img)
    return images

def createMaskAndGetMatch(mask, matcher) :
    total_match = 0
    for i, (fm, sm) in enumerate(matcher) :
        if fm.distance < sm.distance * 0.7 :
            mask[i] = [1,0]
            total_match += 1
            
    return mask, total_match

def SIFT_fm(obj, obj_scn) :
    obj_sift = obj.copy()
    obj_scn_sift = obj_scn.copy()

    SIFT = cv2.SIFT_create()
    
    obj_kp, obj_ds = SIFT.detectAndCompute(obj_sift, None)
    obj_scn_kp, obj_scn_ds = SIFT.detectAndCompute(obj_scn_sift, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm = 1), dict(checks = 50))
    matcher = flann.knnMatch(obj_ds, obj_scn_ds, 2)
    
    mask = [[0,0] for i in range(len(matcher))]
    mask, totalMatch = createMaskAndGetMatch(mask, matcher)
    
    result = cv2.drawMatchesKnn(
        img1=obj_sift,
        img2=obj_scn_sift,
        keypoints1=obj_kp,
        keypoints2=obj_scn_kp,
        matchColor=[0,0,255],
        outImg=None,
        singlePointColor=[255, 0 ,0],
        matchesMask=mask,
        matches1to2=matcher
    )
    
    return result, totalMatch

def get_highest_match_img(target, arr_images) :
    max_match = 0
    max_idx = 0
    iterator = 0
    res_arr = []
    
    for i in arr_images :
        res, match = SIFT_fm(target, i)
        res_arr.append(res)
        if max_match <= match :
            max_match = match
            max_idx = iterator
        
        iterator += 1
    
    return max_idx, max_match, res_arr ,res

def main():
    images, labels = read_all_images()
    target = read_target()

    target = equalize_image(target)
    
    equalized_images = equalize_all_images(images)

    idx, max_match, res_arr, final = get_highest_match_img(target, equalized_images)

    show_stacked_images(res_arr, labels, gray = False)

    show_image(final, f"Best Result : {labels[idx]} | Matches : {max_match}", False)

    pass
    
if __name__ == '__main__':
    main()
    