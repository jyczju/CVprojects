import numpy as np
import cv2
from otsu import otsu,threshold
from dilate_and_erode import dilate,erode

if "__main__" == __name__:
    img = cv2.imread('siling.png', 0) # 读入图片
    origin_img = img.copy()
    cv2.imshow("origin_img", origin_img)

    th = otsu(img, 256) # 使用otsu法确定阈值
    print("otsu法确定的阈值为" + str(th))
    th_img = threshold(img, th)
    cv2.imshow("th_img", th_img)
    
    dilate_result = dilate(th_img, dilate_time=4) # 腐蚀
    cv2.imshow("dilate_result", dilate_result)
    erode_result = erode(dilate_result, erode_time=4) # 膨胀
    cv2.imshow("erode_result", erode_result)

    
    cv2.waitKey(0)