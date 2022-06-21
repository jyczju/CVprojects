import cv2
import numpy as np
from scipy.ndimage import filters

def plane_sweep_gauss(img_l, img_r, start, steps, size):
    '''
    采用平面滑动的方式、使用带高斯加权的归一化相关性指标计算图像的视差
    输入：左右图像，起始视差，滑动步长，滑动窗口大小
    输出：视差矩阵
    '''
    m, n = img_l.shape
    # 计算等效高斯sigma值
    sigma = (size-1)/8

    # 初始化各数组
    mean_l = np.zeros((m, n))
    mean_r = np.zeros((m, n))
    s = np.zeros((m, n))
    s_l = np.zeros((m, n))
    s_r = np.zeros((m, n))

    # 保存深度信息的数组
    depthmaps = np.zeros((m, n, steps))

    # 计算平均值
    filters.gaussian_filter(img_l, sigma, 0, mean_l)
    filters.gaussian_filter(img_r, sigma, 0, mean_r)

    # 归一化图像
    norm_l = img_l - mean_l
    norm_r = img_r - mean_r

    # 遍历范围内的所有视差值
    for dis in range(steps):
        # 计算NCC
        filters.gaussian_filter(np.roll(norm_l, - dis - start) * norm_r, sigma, 0, s)  # 和归一化
        filters.gaussian_filter(np.roll(norm_l, - dis - start) * np.roll(norm_l, - dis - start), sigma, 0, s_l)
        filters.gaussian_filter(norm_r * norm_r, sigma, 0, s_r)  # 和反归一化
        depthmaps[:, :, dis] = s / np.sqrt(s_l * s_r)
    
    # 为每个点选取最佳匹配点并取其视差
    return np.argmax(depthmaps, axis=2) + start

if __name__ == '__main__':
    # 读取图像
    img_l = cv2.imread("tsukuba_l.png",0)
    img_r = cv2.imread("tsukuba_r.png",0)

    # 设置大概的视差范围[start, start+steps-1]
    steps = 15
    start = 3
    
    # NCC窗口大小
    window_size = 9
    
    # 计算视差图像
    res = plane_sweep_gauss(img_l, img_r, start, steps, window_size)
    print(res)

    # 绘制伪彩色视差图
    res_plot = np.array((res-start)/(steps-1.0)*255.0, dtype=np.uint8)
    res_plot =  cv2.applyColorMap(res_plot, cv2.COLORMAP_JET)
    cv2.imshow("depth", res_plot)
    cv2.imwrite("depth_wd"+str(window_size)+".png", res_plot)

    cv2.waitKey(0)
