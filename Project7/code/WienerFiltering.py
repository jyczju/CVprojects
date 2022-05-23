'''
图像复原
浙江大学控制学院《数字图像处理与机器视觉》第五次作业
jyczju
2022/5/14 v1.0
'''
import math
import numpy as np
import cv2

def cal_PSF(img_size, length, angle):
    '''
    计算点扩展函数
    输入：图像尺寸，运动长度，运动角度(弧度制)
    输出：点扩展函数
    '''
    PSF = np.zeros(img_size)  # 点扩展函数初始化
    h,w = img_size
    x_center = int((h - 1) / 2)
    y_center = int((w - 1) / 2) # 图像中心坐标

    # 将angle角度上length个点置成1
    for i in range(length):
        delta_x = round(math.sin(angle) * i)
        delta_y = round(math.cos(angle) * i)
        PSF[int(x_center - delta_x), int(y_center + delta_y)] = 1

    cv2.imwrite('PSF.png', PSF*255)
    PSF = PSF / np.sum(PSF) # 归一化
    return PSF

def wiener_filter(g, PSF, K=0.01):
    '''
    维纳滤波
    输入：模糊图像g，点扩展函数PSF，噪声抑制因子K
    输出：清晰复原图
    '''
    G = np.fft.fft2(g) # 傅里叶变换，计算模糊图频谱
    H = np.fft.fft2(PSF) # 傅里叶变换，计算模糊核频谱
    wiener_fft = np.conj(H) / (np.abs(H) ** 2 + K) # 维纳滤波器公式
    F = wiener_fft * G
    f = np.fft.ifftshift(np.fft.ifft2(F)) # 傅里叶逆变换得到复原后图像
    return f.real

if __name__ == '__main__':
    g = cv2.imread("origin_img.bmp",0) # 读取模糊图像
    length = 30 # PSF长度
    angle = 11 # PSF角度
    angle = angle * math.pi / 180 # 角度转弧度

    PSF = cal_PSF(g.shape, length, angle) # 计算点扩展函数
    f = wiener_filter(g, PSF, K = 0.02) # 维纳滤波
    f = f.astype(np.uint8) # 转换数据类型

    cv2.imwrite("restoreImage.png",f)

    length = 15 # PSF长度
    angle = 11 # PSF角度
    angle = angle * math.pi / 180 # 角度转弧度

    PSF = cal_PSF(g.shape, length, angle) # 计算点扩展函数
    f = wiener_filter(g, PSF, K = 0.02) # 维纳滤波
    f = f.astype(np.uint8) # 转换数据类型

    cv2.imwrite("restoreImage1.png",f)

    length = 30 # PSF长度
    angle = 30 # PSF角度
    angle = angle * math.pi / 180 # 角度转弧度

    PSF = cal_PSF(g.shape, length, angle) # 计算点扩展函数
    f = wiener_filter(g, PSF, K = 0.02) # 维纳滤波
    f = f.astype(np.uint8) # 转换数据类型

    cv2.imwrite("restoreImage2.png",f)
    cv2.waitKey(0)
    cv2.waitKey(0)