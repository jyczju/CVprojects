'''
对高斯噪声和椒盐噪声进行均值滤波、中值滤波和双边滤波
浙江大学控制学院《数字图像处理与机器视觉》第一次上机实践
jyczju
2022/3/18 v1.0
'''
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def add_noise_Guass(img, mean=0, var=30**2):
    '''
    添加高斯噪声
    输入：灰度图像，高斯噪声均值，高斯噪声方差
    输出：添加高斯噪声后的图像
    '''
    noise = np.random.normal(mean, var ** 0.5, img.shape) # 生成噪声
    img = img + noise # 加入噪声
    img_guass = np.clip(img, 0, 255) # 防止值超限
    img_guass = np.uint8(img_guass)
    return img_guass


def add_noise_SP(img, SNR=0.6):
    '''
    添加椒盐噪声
    输入：灰度图像，信噪比（0~1之间）
    输出：添加椒盐噪声后的图像
    '''
    SP = int((img.shape[0]*img.shape[1])*(1-SNR)) # 计算椒盐噪声个数
    for i in range(SP):
        randx=np.random.randint(1,img.shape[0]-1)   # 生成一个 1 至 hight-1 之间的随机整数
        randy=np.random.randint(1,img.shape[1]-1)   # 生成一个 1 至 width-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            img[randx,randy]=0
        else:
            img[randx,randy]=255
    return img

def mean_Filter(img, size=(5,5)):
    '''
    均值滤波
    输入：灰度图像，模板尺寸
    输出：均值滤波后的图像
    '''
    kernal = np.ones(size, np.float32)/size[0]/size[1] # 模板
    h = img.shape[0]
    w = img.shape[1]
    for x in range(int((size[0]-1)/2),int(h-(size[0]-1)/2)):
        for y in range(int((size[1]-1)/2),int(w-(size[1]-1)/2)):
            window = img[int(x-(size[0]-1)/2):int(x+(size[0]-1)/2+1),int(y-(size[1]-1)/2):int(y+(size[1]-1)/2+1)] # 当前操作窗口
            img[x][y] = np.sum(window*kernal) # 对各点进行模板操作
    return img

def mid_Filter(img, size=(5,5)):
    '''
    中值滤波
    输入：灰度图像，模板尺寸
    输出：中值滤波后的图像
    '''
    h = img.shape[0]
    w = img.shape[1]
    for x in range(int((size[0]-1)/2),int(h-(size[0]-1)/2)):
        for y in range(int((size[1]-1)/2),int(w-(size[1]-1)/2)):
            window = img[int(x-(size[0]-1)/2):int(x+(size[0]-1)/2+1),int(y-(size[1]-1)/2):int(y+(size[1]-1)/2+1)] # 当前操作窗口
            img[x][y] = np.median(window) # 对各点进行模板操作
    return img

    
def bf_Filter(img, size=(5,5), sigma_s = 10, sigma_r = 10):
    '''
    双边滤波
    输入：灰度图像，模板尺寸，空间域方差，像素值域方差
    输出：双边滤波后的图像
    '''
    kernal = np.ones(size, np.float32) # 模板初始化
    h = img.shape[0]
    w = img.shape[1]
    for x in range(int((size[0]-1)/2),int(h-(size[0]-1)/2)):
        for y in range(int((size[1]-1)/2),int(w-(size[1]-1)/2)):
            window = img[int(x-(size[0]-1)/2):int(x+(size[0]-1)/2+1),int(y-(size[1]-1)/2):int(y+(size[1]-1)/2+1)] # 当前操作窗口
            for i in range(size[0]):
                for j in range(size[1]):
                    dist = ((i-(size[0]-1)/2)**2+(j-(size[0]-1)/2)**2)/2/sigma_s
                    dValue = ((float(window[i][j]) - float(window[int((size[0]-1)/2)][int((size[1]-1)/2)]))**2)/2/sigma_r
                    kernal[i][j] = math.exp(-dist)*math.exp(-dValue) # 计算模板权重
            kernal /= np.sum(kernal) # 权重归一化
            img[x][y] = np.sum(window*kernal) # 对各点进行模板操作
    return img


if __name__ == '__main__':

    # 读取原始图像
    img = cv2.imread('GrayPhoto.jpg', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("img", img)

    # 对原图像添加高斯噪声
    img_guass = add_noise_Guass(img, 0, 30**2)
    cv2.imshow("img_guass", img_guass)
    cv2.imwrite('img_guass.jpg', img_guass)

    # 对原图像添加椒盐噪声
    img_SP = add_noise_SP(img, 0.92)
    cv2.imshow("img_SP", img_SP)
    cv2.imwrite('img_SP.jpg', img_SP)

    # 均值滤波
    img_mean_guass_cv = cv2.blur(img_guass, (5,5))
    cv2.imshow("img_mean_guass_cv", img_mean_guass_cv)
    cv2.imwrite("img_mean_guass_cv.jpg", img_mean_guass_cv)
    img_mean_guass = mean_Filter(img_guass, (5,5))
    cv2.imshow("img_mean_guass", img_mean_guass)
    cv2.imwrite("img_mean_guass.jpg", img_mean_guass)

    img_mean_SP_cv = cv2.blur(img_SP, (5,5))
    cv2.imshow("img_mean_SP_cv", img_mean_SP_cv)
    cv2.imwrite("img_mean_SP_cv.jpg", img_mean_SP_cv)
    img_mean_SP = mean_Filter(img_SP, (5,5))
    cv2.imshow("img_mean_SP", img_mean_SP)
    cv2.imwrite("img_mean_SP.jpg", img_mean_SP)


    # 中值滤波
    img_mid_guass_cv=cv2.medianBlur(img_guass,5)
    cv2.imshow("img_mid_guass_cv", img_mid_guass_cv)
    cv2.imwrite("img_mid_guass_cv.jpg", img_mid_guass_cv)
    img_mid_guass=mid_Filter(img_guass,(3,3) )
    cv2.imshow("img_mid_guass", img_mid_guass)
    cv2.imwrite("img_mid_guass.jpg", img_mid_guass)

    img_mid_SP_cv=cv2.medianBlur(img_SP,3)
    cv2.imshow("img_mid_SP_cv", img_mid_SP_cv)
    cv2.imwrite("img_mid_SP_cv.jpg", img_mid_SP_cv)
    img_mid_SP=mid_Filter(img_SP,(3,3) )
    cv2.imshow("img_mid_SP", img_mid_SP)
    cv2.imwrite("img_mid_SP.jpg", img_mid_SP)


    # 双边滤波
    img_bf_guass=bf_Filter(img_guass,(3,3),100,100**2)
    cv2.imshow("img_bf_guass", img_bf_guass)
    cv2.imwrite("img_bf_guass.jpg", img_bf_guass)
    img_bf_guass_cv=cv2.bilateralFilter(img_guass,3,100,15)
    cv2.imshow("img_bf_guass_cv", img_bf_guass_cv)
    cv2.imwrite("img_bf_guass_cv.jpg", img_bf_guass_cv)

    img_bf_SP=bf_Filter(img_SP,(3,3),100,150**2)
    cv2.imshow("img_bf_SP", img_bf_SP)
    cv2.imwrite("img_bf_SP.jpg", img_bf_SP)
    img_bf_SP_cv=cv2.bilateralFilter(img_SP,3,100,15)
    cv2.imshow("img_bf_SP_cv", img_bf_SP_cv)
    cv2.imwrite("img_bf_SP_cv.jpg", img_bf_SP_cv)

    K = cv2.waitKey(0)