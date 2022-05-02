from cv2 import imwrite
from SplitMerge import extract_contour,find_cnts
import cv2
import numpy as np

if __name__ == '__main__':

    img = cv2.imread('contour_img.png',0) # 70,328
    # print(img.shape)
    # img = np.array(img)
    
    region_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1] # 对图像进行二值化
    for i in range(70-5,70+6):
        for j in range(328-5,328+6):
            region_img[i][j] = 0

    cv2.imshow('img', region_img)

    
    img1 = extract_contour(region_img)
    cv2.imshow('img1', img1)
    cv2.imwrite('img1.png',img1)

    freeman = find_cnts(img1)
    print(freeman)

    cv2.waitKey(0)