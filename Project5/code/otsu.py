'''
otsu法二值化
浙江大学控制学院《数字图像处理与机器视觉》第四次作业
jyczju
2022/4/17 v1.0
'''
import numpy as np

def otsu(img, GrayScale):
    '''
    otsu法确定二值化阈值
    输入：图像，灰度级
    输出：二值化阈值
    '''
    pixel_sum=img.shape[0]*img.shape[1] # 总像素数目初始化
    hist=np.zeros(GrayScale) # 各个灰度值像素数目初始化
    
    w=np.zeros(GrayScale) # 出现概率值函数w(k)
    u=np.zeros(GrayScale) # 平均灰度值函数u(k)
    
    # 统计各个灰度值的像素个数
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i][j]] += 1 # 当前灰度值数目+1

    # 计算w(k)
    for i in range(GrayScale):
        w[i]=np.sum(hist[:i])*1.0/pixel_sum

    # 计算u(k)
    for i in range(GrayScale):
        sum_temp = 0
        for j in range(i+1):
            sum_temp += hist[j]*j
        u[i]=sum_temp*1.0/pixel_sum
      
    # 确定最大类间方差对应的阈值
    Max_var = 0 # 最大类间方差初始化
    for thi in range(1, GrayScale): # 遍历每一个阈值
        w1=w[thi]-w[0] # 前景像素的比例
        w2=w[-1]-w[thi] # 背景像素的比例

        if w1 != 0 and w2 != 0: # 确保分母不为0
            u1 = (u[thi]-u[0]) * 1.0 / w1 # 前景像素的平均灰度值
            u2 = (u[-1]-u[thi])* 1.0 / w2 # 背景像素的平均灰度值
            tem_var=w1*np.power((u1-u[-1]),2)+w2*np.power((u2-u[-1]),2) # 当前类间方差
            if Max_var<tem_var: # 判断当前类间方差是否为最大值
                Max_var=tem_var # 更新最大值
                th=thi # 更新阈值
    return th 

def threshold(img, th):
    '''
    图像二值化
    输入：灰度图像，阈值
    输出：二值化后的图像
    '''
    th_img = np.zeros(img.shape, dtype=np.uint8) # 二值化图像初始化
    th_img[img>=th] = 255 # 将大于阈值的像素置为255
    th_img[img<th] = 0 # 将小于阈值的像素置为0
    return th_img

