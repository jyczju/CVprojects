import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#1.读入图片
img = cv.imread('ex_red_3.jpg')
gray_img = img

#2.canny边缘检测
canny = cv.Canny(gray_img,80,150)

#3.轮廓检测并设置标记图像
#寻找图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
contours,hierarchy = cv.findContours(canny,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#32位有符号整数类型，
marks = np.zeros(img.shape[:2],np.int32)
#findContours检测到的轮廓
imageContours = np.zeros(img.shape[:2],np.uint8)

#轮廓颜色
compCount = 0
index = 0
#绘制每一个轮廓
for index in range(len(contours)):
    #对marks进行标记，对不同区域的轮廓使用不同的亮度绘制，相当于设置注水点，有多少个轮廓，就有多少个轮廓
    #图像上不同线条的灰度值是不同的，底部略暗，越往上灰度越高
    marks = cv.drawContours(marks,contours,index,(index,index,index),1,8,hierarchy)
    #绘制轮廓，亮度一样
    imageContours = cv.drawContours(imageContours,contours,index,(255,255,255),1,8,hierarchy)  


#4 使用分水岭算法，并给不同的区域随机填色
marks = cv.watershed(img,marks)
afterWatershed = cv.convertScaleAbs(marks)  

#生成随机颜色
colorTab = np.zeros((np.max(marks)+1,3))
#生成0~255之间的随机数
for i in range(len(colorTab)):
    aa = np.random.uniform(0,255)
    bb = np.random.uniform(0,255)
    cc = np.random.uniform(0,255)
    colorTab[i] = np.array([aa,bb,cc],np.uint8)

bgrImage = np.zeros(img.shape,np.uint8)

#遍历marks每一个元素值，对每一个区域进行颜色填充
for i in range(marks.shape[0]):
    for j in range(marks.shape[1]):
        #index值一样的像素表示在一个区域
        index = marks[i][j]
        #判断是不是区域与区域之间的分界,如果是边界(-1)，则使用白色显示
        if index == -1:
            bgrImage[i][j] = np.array([255,255,255])
        else:                        
            bgrImage[i][j]  = colorTab[index]
# 5 图像显示
plt.imshow(bgrImage[:,:,::-1])
plt.title('图像分割结果')
plt.xticks([]), plt.yticks([])
plt.show()