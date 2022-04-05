import numpy as np
import cv2 
import matplotlib.pyplot as plt # plt 用于显示图片


#判断方框是否需要再次拆分为四个
def judge(w0, h0, w, h):
    a = img[h0: h0 + h, w0: w0 + w]
    ave = np.mean(a)
    std = np.std(a, ddof=1)
    count = 0
    total = 0
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
        #注意！我输入的图片数灰度图，所以直接用的img[j,i]，RGB图像的话每个img像素是一个三维向量，不能直接与avg进行比较大小。
            if abs(img[j, i] - ave) < 1 * std:
                count += 1
            total += 1
    if (count / total) < 0.95:#合适的点还是比较少，接着拆
        return True
    else:
        return False

##将图像将根据阈值二值化处理，在此默认125
def draw(w0, h0, w, h):
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if img[j, i] > 125:
                img[j, i] = 255
            else:
                img[j, i] = 0


def function(w0, h0, w, h):
    if judge(w0, h0, w, h) and (min(w, h) > 5):
        function(w0, h0, int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0, int(w / 2), int(h / 2))
        function(w0, h0 + int(h / 2), int(w / 2), int(h / 2))
        function(w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
    else:
        draw(w0, h0, w, h)

##############################################
######################main_##################
###############################################

img = cv2.imread('rice.png', 0)
img_input = cv2.imread('rice.png', 0)#备份

height, width = img.shape

function(0, 0, width, height)

cv2.imshow('input',img_input)
cv2.imshow('output',img)

cv2.waitKey()
cv2.destroyAllWindows()

