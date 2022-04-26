'''
图像切割+边缘识别
浙江大学控制学院《数字图像处理与机器视觉》第三次作业
jyczju
2022/4/6 v1.0
'''
import cv2
import numpy as np
import sys
from imgnode import ImgNode
import matplotlib.pyplot as plt

def region_split_merge(img, min_area=(1,1), threshold=5.0):
    '''
    区域分裂合并算法，主要依靠ImgNode类实现
    输入：待处理图像，分裂的最小区域，合并的相似性判断阈值
    输出：前后景分割后的二值化图像
    '''
    draw_img = img.copy() # 用于绘制分裂结果的图像

    start_node = ImgNode(img, None, 0, img.shape[0], 0, img.shape[1]) # 创建起始节点，即整幅图像

    draw_img = start_node.split(draw_img, min_area) # 区域分裂

    leaf_father = start_node.find_leaf_father() # 寻找开始合并的节点
    region_img = np.zeros((int(img.shape[0]), int(img.shape[1]))) # 二值化图像初始化
    region_img = leaf_father.sub_node3.merge(region_img, threshold) # 区域合并

    return region_img,draw_img

def extract_contour(region_img):
    '''
    轮廓提取，某一像素周边若有背景像素，则认为其为轮廓
    输入：二值化图像，目标像素为黑色，背景像素为白色
    输出：轮廓图像，轮廓为黑色，背景为白色
    '''
    contour_img = region_img.copy() # 初始化轮廓图像

    for h in range(1,region_img.shape[0]-1):
        for w in range(1,region_img.shape[1]-1): # 遍历图像中的每一点
            if region_img[h][w] == 0 and region_img[h-1][w] == 0 and region_img[h+1][w] == 0 and region_img[h][w-1] == 0 and region_img[h][w+1] == 0:
                contour_img[h][w] = 255 # 若像素本身及其周围像素均为黑色，则其为内部点，将其置为白色

    return contour_img

def track_contour(img, start_point):
    '''
    轮廓跟踪
    输入：边界图像，当前轮廓起始点
    输出：当前轮廓轮廓链码
    '''
    neibor = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]  # 8连通方向码
    dir = 5  # 起始搜索方向
    contours = [start_point]  # 用于存储轮廓点

    current_point = start_point # 将轮廓的开始点设为当前点
    
    neibor_point = tuple(np.array(current_point) + np.array(neibor[dir])) # 通过当前点和邻域点集以及链码值确定邻点

    if neibor_point[0] >= img.shape[0] or neibor_point[1] >= img.shape[1] or neibor_point[0] < 0 or neibor_point[1] < 0: # 若邻点超出边界，则轮廓结束
        return contours

    while True:  # 轮廓扫描循环
        # print('current_point',current_point)
        while img[neibor_point[0], neibor_point[1]] != 0:  # 邻点不是边界点
            dir += 1  # 逆时针旋转45度进行搜索
            if dir >= 8:
                dir -= 8
            neibor_point = tuple(np.array(current_point) + np.array(neibor[dir])) # 更新邻点

            if neibor_point[0] >= img.shape[0] or neibor_point[1] >= img.shape[1] or neibor_point[0] < 0 or neibor_point[1] < 0: # 若邻点超出边界，则轮廓结束
                return contours

        else:  
            current_point = neibor_point # 将符合条件的邻域点设为当前点进行下一次的边界点搜索
            contours.append(current_point) # 将当前点加入轮廓list
            if (dir % 2) == 0:
                dir += 7
            else:
                dir += 6
            if dir >= 8:
                dir -= 8 # 更新方向
            neibor_point = tuple(np.array(current_point) + np.array(neibor[dir])) # 更新邻点

            if neibor_point[0] >= img.shape[0] or neibor_point[1] >= img.shape[1] or neibor_point[0] < 0 or neibor_point[1] < 0: # 若邻点超出边界，则轮廓结束
                return contours

        if current_point == start_point:
            break # 当搜索点回到起始点，搜索结束，退出循环

    return contours

def draw_contour(img, contours, color=(0, 0, 255)):
    '''
    在img上绘制轮廓
    输入：欲绘制的图像，轮廓链码，颜色
    输出：绘制好的图像
    '''
    for (x, y) in (contours): # 绘制轮廓
        img[x-1:x+1, y-1:y+1] = color  # 粗
        # img_cnt[x,y] = (0, 0, 255) # 细
    return img

def find_start_point(img, all_cnts):
    '''
    寻找起始点
    输入：边界图像，已被识别到的轮廓list
    输出：起始点
    '''
    start_point = (-1, -1) # 初始化起始点

    # 寻找起始点
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0 and (i, j) not in all_cnts: # 点为黑色且不在已识别到的轮廓list中
                start_point = (i, j) # 找到新的起始点
                break
        if start_point != (-1, -1):
            break
    return start_point

def find_cnts(img):
    '''
    寻找轮廓集合
    输入：边界图像
    输出：轮廓集合（list,每一项都是一个轮廓链码）
    '''
    contours = [] # 当前边界轮廓初始化
    cnts = [] # 轮廓集合初始化
    all_cnts = [] # 所有已找到的轮廓点

    while True:
        start_point = find_start_point(img, all_cnts) # 寻找当前边界的轮廓起始点
        # print('start point:', start_point)

        if start_point == (-1, -1): # 若找不到新的起始点，则说明所有的轮廓点都已被找到，退出循环
            break

        contours = track_contour(img, start_point) # 寻找当前边界的轮廓
        cnts.append(contours) # 将找到的轮廓加入轮廓集合中
        all_cnts = all_cnts + contours # 将找到的轮廓点加入轮廓点集合中

    return cnts

def draw_cnts(cnts, img, color=(0, 0, 255)):
    '''
    绘制所有轮廓
    输入：轮廓集合，欲绘制的图像，颜色
    输出：绘制好的图像
    '''
    for cnt in cnts:
        img = draw_contour(img, cnt, color) # 逐一绘制每个轮廓
    return img

def contours_filter(cnts, windows_size = 13):
    '''
    对轮廓进行滤波（均值滤波）
    输入：轮廓集合，滤波窗口大小
    输出：滤波后的轮廓集合
    '''
    if (windows_size % 2) == 0:
        windows_size += 1 # 保证windows_size为奇数
    cnts_filter = [] # 初始化滤波后的轮廓集合
    for cnt in cnts:
        for i in range(int((windows_size-1)/2), len(cnt)-int((windows_size-1)/2)):
            ix = np.mean([cnt[j][0] for j in range(i-int((windows_size-1)/2), i+int((windows_size-1)/2)+1)])
            iy = np.mean([cnt[j][1] for j in range(i-int((windows_size-1)/2), i+int((windows_size-1)/2)+1)])
            cnt[i] = (int(ix), int(iy)) # 均值滤波
        cnts_filter.append(cnt) # 将滤波后的轮廓添加到集合中
    return cnts_filter


if __name__ == '__main__':
    sys.setrecursionlimit(100000) # 设置最大允许递归深度

    # img = cv2.imread('zjui_logo.png', 0) # 读入图像
    img = cv2.imread('zju_logo.png', 0) # 读入图像
    # img = cv2.imread('zju_logo_gauss.png', 0) # 读入图像
    # img = cv2.imread('zjui_logo_gauss.png', 0) # 读入图像
    # img = cv2.imread('zju_logo_uneven.png', 0) # 读入图像
    # img = cv2.imread('zjui_logo_uneven.png', 0) # 读入图像
    origin_img = img.copy() # 备份原始图像
    # cv2.imshow('origin_img', origin_img)

    region_img,draw_img = region_split_merge(img, min_area=(1,1), threshold=5.0) # 5.0 # 区域分裂合并
    # cv2.imshow('draw_img', draw_img) # 显示区域分裂结果
    cv2.imwrite('draw_img.png', draw_img)

    # cv2.imshow('region_img', region_img) # 显示区域合并结果
    region_img[275:300, 0:450] = 255 # 将区域图像中的一部分置为白色
    region_img[0:300, 445:450] = 255 # 将区域图像中的一部分置为白色


    cv2.imwrite('region_img.png', region_img)

    contour_img = extract_contour(region_img) # 轮廓提取
    # cv2.imshow('contour_img', contour_img) # 显示轮廓图像
    cv2.imwrite('contour_img.png', contour_img) 

    cnts = find_cnts(contour_img) # 轮廓跟踪
    print('cnts:')
    print(cnts)

    # img_cnt = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
    img_cnt = 255*np.ones([img.shape[0], img.shape[1], 3])
    img_cnt = draw_cnts(cnts, img_cnt, color = (0, 0, 255)) # 绘制轮廓跟踪结果
    # cv2.imshow('img_cnt', img_cnt)
    cv2.imwrite('img_cnt.png', img_cnt)

    cnts_filter = contours_filter(cnts, windows_size = 11) # 轮廓链码滤波
    print('cnts_filter')
    print(cnts_filter)

    # img_cnt_filter = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
    img_cnt_filter = 255*np.ones([img.shape[0], img.shape[1], 3])
    img_cnt_filter = draw_cnts(cnts_filter, img_cnt_filter, color=(255, 0, 0)) # 绘制轮廓链码滤波结果
    # cv2.imshow('img_cnt_filter', img_cnt_filter)
    cv2.imwrite('img_cnt_filter.png', img_cnt_filter)    

    plt.figure(figsize=(9, 9.5))
    title_size = 12
    plt.subplot(321)
    plt.axis('off')
    plt.imshow(origin_img,cmap='gray')
    plt.title("Figure 1: Original image",fontdict={'weight':'normal','size': title_size})

    plt.subplot(322)
    plt.axis('off')
    plt.imshow(draw_img,cmap='gray')
    plt.title("Figure 2: Splited image",fontdict={'weight':'normal','size': title_size})

    plt.subplot(323)
    plt.axis('off')
    plt.imshow(region_img,cmap='gray')
    plt.title("Figure 3: Merged image",fontdict={'weight':'normal','size': title_size})

    plt.subplot(324)
    plt.axis('off')
    plt.imshow(contour_img,cmap='gray')
    plt.title("Figure 4: Contours",fontdict={'weight':'normal','size': title_size})

    plt.subplot(325)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_cnt.astype(np.float32),cv2.COLOR_BGR2RGB))
    plt.title("Figure 5: Contours tracked by ChainCode",fontdict={'weight':'normal','size': title_size})

    plt.subplot(326)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(img_cnt_filter.astype(np.float32),cv2.COLOR_BGR2RGB))
    plt.title("Figure 6: Filtered Contours",fontdict={'weight':'normal','size': title_size})

    plt.show()

    cv2.waitKey(0)
