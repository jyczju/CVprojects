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
    生成运动模糊核
    输入：图像尺寸，运动长度，运动角度(弧度)
    输出：运动模糊核
    '''
    PSF = np.zeros(img_size)  # 运动模糊核初始化
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
    输入：模糊图像g，运动模糊核PSF，噪声抑制因子K
    输出：清晰复原图
    '''
    G = np.fft.fft2(g) # 傅里叶变换，计算模糊图频谱
    H = np.zeros(G.shape)+1j # 运动模糊核频谱初始化
    a = 30*math.cos(11/180*math.pi)
    b = 30*math.sin(11/180*math.pi)
    T = 12000000
    for u,v in np.ndindex(G.shape):
        # u -= 127
        # v -= 127
        if u == 0 and v == 0:
            H[u,v] = 1
        else:
            H[u,v] = T*1j/2/math.pi/(a*(u)+b*(v))*(math.e**(-2j*math.pi*(a*(u)+b*(v)))-1)

    h = np.fft.ifftshift(np.fft.ifft2(H)) # 傅里叶逆变换

    # print(H)

    h_img = h.real-3.5
    h_img = h_img.astype(np.uint8)

    print(h_img)
    cv2.imshow('h_img', h_img)

    wiener_fft = np.conj(H) / (np.abs(H) ** 2 + K) # 维纳滤波器公式
    f = np.fft.ifftshift(np.fft.ifft2(wiener_fft * G)) # 傅里叶逆变换得到复原后图像
    return f.real

if __name__ == '__main__':
    g = cv2.imread("origin_img.bmp",0) # 读取模糊图像
    length = 30 # PSF长度
    angle = 11 # PSF角度
    angle = angle * math.pi / 180 # 角度转弧度

    PSF = cal_PSF(g.shape, length, angle) # 生成生成运动模糊核
    f = wiener_filter(g, PSF, K = 1/60) # 维纳滤波
    f = f.astype(np.uint8) # 转换数据类型

    cv2.imwrite("restoreImage.png",f)
    cv2.waitKey(0)