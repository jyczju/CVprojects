'''
图像细化
浙江大学控制学院《数字图像处理与机器视觉》第四次作业
jyczju
2022/4/17 v1.0
'''
import numpy as np
import cv2

def img_thin(img):
    '''
    图像细化，提取骨架
    输入：二值化图像
    输出：细化后的图像
    '''
    h, w = img.shape # 获取图像的高和宽

    out = np.zeros((h, w), dtype=int) # 初始化输出图像
    out[img > 0] = 1 # 将二值化图像转换为1和0
    out = 1-out # 取反

    i = 0 # 调试用  可以查看细化的过程
    while True:
        delet_node1 = []
        delet_node2 = [] # 创建删除点list

        # step 1
        for x in range(1, h-1):
            for y in range(1, w-1):
                if out[x, y] == 1: # 如果是前景点
                    num_of_one = np.sum(out[x-1:x+2, y-1:y+2])-1 # p1的非零邻点个数
                    if num_of_one >= 2 and num_of_one <= 6: # 如果p1的非零邻点个数在2到6之间
                        if count_0_to_1(out, x, y) == 1: # 若从0到1的变化次数为1
                            if out[x-1, y]*out[x+1, y]*out[x, y+1] == 0 and out[x, y-1]*out[x+1, y]*out[x, y+1] == 0:
                                    delet_node1.append((x, y)) # 将当前点加入删除点list

        for node in delet_node1: # 对删除点list进行遍历
            out[node] = 0 # 将删除点设置为0

        # step 2
        for x in range(1, h-1):
            for y in range(1, w-1):
                if out[x, y] == 1: # 如果是前景点
                    num_of_one = np.sum(out[x-1:x+2, y-1:y+2])-1 # p1的非零邻点个数
                    if num_of_one >= 2 and num_of_one <= 6: # 如果p1的非零邻点个数在2到6之间
                        if count_0_to_1(out, x, y) == 1: # 若从0到1的变化次数为1
                            if out[x-1, y]*out[x, y-1]*out[x+1, y] == 0 and out[x-1, y]*out[x, y-1]*out[x, y+1] == 0:
                                    delet_node2.append((x, y)) # 将当前点加入删除点list
        
        for node in delet_node2: # 对删除点list进行遍历
            out[node] = 0 # 将删除点设置为0
        
        # 调试用  可以查看细化的过程
        tmp = out.copy()
        tmp = 1-tmp
        tmp = tmp.astype(np.uint8) * 255
        cv2.imwrite('./results/'+str(i)+'.png', tmp)
        i += 1

        if len(delet_node1) ==0 and len(delet_node2) == 0: # 如果没有点再满足标记删除的条件，则退出循环
            break

    out = 1 - out # 取反
    out = out.astype(np.uint8) * 255 # 转换为0和255

    return out

def count_0_to_1(img, x, y):
    '''
    计算从0到1的变化次数
    输入：二值化图像，坐标
    输出：从0到1的变化次数
    '''
    num = 0
    if (img[x-1, y-1] - img[x-1, y]) == 1: # p2到p3
        num += 1
    if (img[x, y-1] - img[x-1, y-1]) == 1: # p3到p4
        num += 1
    if (img[x+1, y-1] - img[x, y-1]) == 1: # p4到p5
        num += 1
    if (img[x+1, y] - img[x+1, y-1]) == 1: # p5到p6
        num += 1
    if (img[x+1, y+1] - img[x+1, y]) == 1: # p6到p7
        num += 1
    if (img[x, y+1] - img[x+1, y+1]) == 1: # p7到p8
        num += 1
    if (img[x-1, y+1] - img[x, y+1]) == 1: # p8到p9
        num += 1
    if (img[x-1, y] - img[x-1, y+1]) == 1: # p9到p2
        num += 1
    return num

