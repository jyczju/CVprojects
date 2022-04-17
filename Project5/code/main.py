'''
主程序，实现了otsu法二值化以及形态学滤波，并进行图像细化
浙江大学控制学院《数字图像处理与机器视觉》第四次作业
jyczju
2022/4/17 v1.0
！！！请运行本程序！！！
'''
import cv2
from otsu import otsu,threshold
from dilate_and_erode import dilate,erode
from thin import img_thin

if "__main__" == __name__:
    img = cv2.imread('siling3.png', 0) # 读入图片
    origin_img = img.copy() # 备份原图
    cv2.imshow("origin_img", origin_img) # 显示原图

    th = otsu(img, 256) # 使用otsu法确定阈值
    print("otsu法确定的阈值为" + str(th))
    th_img = threshold(img, th) # 使用阈值th进行二值化
    cv2.imshow("th_img", th_img) # 显示二值化图像
    cv2.imwrite("th_img.png", th_img) # 保存二值化图像
    
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

    out = img_thin(dilate_result) # 图像细化
    cv2.imshow("thin_result", out)
    cv2.imwrite("thin_result.png", out)
    
    cv2.waitKey(0)