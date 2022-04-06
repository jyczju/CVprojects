import cv2
import numpy as np


def track_contour(img, start_point):
    # 定义链码相对应的增量坐标
    neibor = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
              (0, -1), (1, -1), (1, 0), (1, 1)]  # 邻域点
    dir = 5  # 链码值，也是neibor的索引序号，这里是从链码的5号位开始搜索
    contours = [start_point]  # 用于存储轮廓点

    # 将当前点设为轮廓的开始点
    current_point = start_point

    # dir=5，表示从链码的5方向进行邻域检索，通过当前点和邻域点集以及链码值确定邻域点
    neibor_point = tuple(np.array(current_point) + np.array(neibor[dir]))

    while True:  # 轮廓扫描循环
        # print('current_point',current_point)
        while img[neibor_point[0], neibor_point[1]] != 0:  # 临点不是边界点
            dir += 1  # 逆时针旋转45度进行搜索
            if dir >= 8:
                dir -= 8
            neibor_point = tuple(
                np.array(current_point) + np.array(neibor[dir]))
        else:
            # 将符合条件的邻域点设为当前点进行下一次的边界点搜索
            current_point = neibor_point
            contours.append(current_point)
            if (dir % 2) == 0:
                dir += 7
            else:
                dir += 6
            if dir >= 8:
                dir -= 8
            neibor_point = tuple(
                np.array(current_point) + np.array(neibor[dir]))

        if current_point == start_point:
            break

    return contours


def draw_contour(img_cnt, contours, color=(0, 0, 255)):
    # 对已经检测到的轮廓进行标记
    for (x, y) in (contours):
        img_cnt[x-1:x+1, y-1:y+1] = color  # 粗
        # img_cnt[x,y] = (0, 0, 255) # 细
    return img_cnt


def find_start_point(img, all_cnts):
    # 初始化起始点
    start_point = (-1, -1)

    # 寻找起始点
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 0 and (i, j) not in all_cnts:
                start_point = (i, j)
                break
        if start_point != (-1, -1):
            break
    return start_point


def find_cnts(img):
    contours = []
    cnts = []
    all_cnts = []

    # for num in range(2):
    while True:
        start_point = find_start_point(img, all_cnts)

        if start_point == (-1, -1):
            break

        # print('start point:', start_point)

        contours = track_contour(img, start_point)
        cnts.append(contours)
        all_cnts = all_cnts + contours
        # print('contours:')
        # print(contours)

    return cnts


def draw_cnts(cnts, img_cnt, color=(0, 0, 255)):
    for cnt in cnts:
        img_cnt = draw_contour(img_cnt, cnt, color)
    return img_cnt


def contours_filter(cnts, size = 13):
    if (size % 2) == 0:
        size += 1
    cnts_filter = []
    for cnt in cnts:
        for i in range(int((size-1)/2), len(cnt)-int((size-1)/2)):
            ix = np.mean([cnt[j][0] for j in range(
                i-int((size-1)/2), i+int((size-1)/2)+1)])
            iy = np.mean([cnt[j][1] for j in range(
                i-int((size-1)/2), i+int((size-1)/2)+1)])

            cnt[i] = (int(ix), int(iy))
        cnts_filter.append(cnt)
    return cnts_filter

if __name__ == '__main__':
    origin_img = cv2.imread("zjui_logo.png", 0)
    img = cv2.imread("contour_img.png", 0)

    # opencv实现
    # contours, _ = cv2.findContours(origin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(contours)
    # img_cnt = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_cnt,contours,-1,(0,0,255),1)
    # cv2.imshow('img_cnt',img_cnt)

    # 自实现
    cnts = find_cnts(img)

    # img_cnt = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
    img_cnt = 255*np.ones([img.shape[0], img.shape[1], 3])
    img_cnt = draw_cnts(cnts, img_cnt, color=(0, 0, 255))

    cv2.imshow('img_cnt', img_cnt)
    cv2.imwrite('img_cnt.png', img_cnt)

    # 链码滤波
    cnts_filter = contours_filter(cnts, size = 13)
    # print(cnts_filter)

    # img_cnt_filter = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)
    img_cnt_filter = 255*np.ones([img.shape[0], img.shape[1], 3])
    img_cnt_filter = draw_cnts(cnts_filter, img_cnt_filter, color=(255, 0, 0))

    cv2.imshow('img_cnt_filter', img_cnt_filter)
    cv2.imwrite('img_cnt_filter.png', img_cnt_filter)

    cv2.waitKey(0)
