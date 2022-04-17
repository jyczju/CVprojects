from unittest import result
import numpy as np
import cv2
from otsu import otsu,threshold
from dilate_and_erode import dilate,erode
from thin import img_thin


if "__main__" == __name__:
    img = cv2.imread('siling3.png', 0) # 读入图片
    origin_img = img.copy()
    cv2.imshow("origin_img", origin_img)

    th = otsu(img, 256) # 使用otsu法确定阈值
    print("otsu法确定的阈值为" + str(th))
    th_img = threshold(img, th)
    cv2.imshow("th_img", th_img)
    cv2.imwrite("th_img.png", th_img)
    
    # 开运算
    dilate_result = dilate(th_img, dilate_time=2) # 腐蚀
    erode_result = erode(dilate_result, erode_time=2) # 膨胀
    cv2.imshow("open_result", erode_result)
    cv2.imwrite("open_result.png", erode_result)

    # 闭运算
    erode_result = erode(erode_result, erode_time=4) # 膨胀
    dilate_result = dilate(erode_result, dilate_time=4) # 腐蚀
    cv2.imshow("close_result", dilate_result)
    cv2.imwrite("close_result.png", dilate_result)

    out = img_thin(dilate_result)
    cv2.imshow("thin_result", out)
    cv2.imwrite("thin_result.png", out)
    
    cv2.waitKey(0)