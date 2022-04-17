'''
形态学滤波（腐蚀与膨胀）
浙江大学控制学院《数字图像处理与机器视觉》第四次作业
jyczju
2022/4/17 v1.0
'''
import numpy as np

def dilate(img, dilate_time=1):
    '''
    对图像进行腐蚀
    输入：二值化图像，腐蚀次数
    输出：腐蚀后的图像
    '''
    h, w = img.shape # 获取图像的高和宽
    kernal = np.array(((0, 1, 0),(1, 0, 1),(0, 1, 0)), dtype=int) # 创建腐蚀核

    out = img.copy() # 创建输出图像
    for i in range(dilate_time):
        tmp = out.copy()
        for x in range(1, h-1):
            for y in range(1, w-1):
                if np.sum(kernal * tmp[x-1:x+2, y-1:y+2]) >= 255: # 至少有一个像素点为背景
                    out[x, y] = 255 # 将中心点置为背景
    return out

def erode(img, erode_time=1):
    '''
    对图像进行膨胀
    输入：二值化图像，膨胀次数
    输出：膨胀后的图像
    '''
    h, w = img.shape # 获取图像的高和宽
    kernal = np.array(((0, 1, 0),(1, 0, 1),(0, 1, 0)), dtype=int) # 创建膨胀核

    out = img.copy() # 创建输出图像
    for i in range(erode_time):
        tmp = out.copy()
        for x in range(1, h-1):
            for y in range(1, w-1):
                if np.sum(kernal * tmp[x-1:x+2, y-1:y+2]) < 255*4: # 至少有一个像素点为前景
                    out[x, y] = 0 # 将中心点置为前景
    return out