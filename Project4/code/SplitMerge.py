import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

i = 0
j = 1
Node_List = []


class ImgNode():
    def __init__(self, img, father_node, h0, h1, w0, w1):
        self.img = img
        self.father_node = father_node
        self.sub_node1 = None
        self.sub_node2 = None
        self.sub_node3 = None
        self.sub_node4 = None
        self.left_node = None
        self.right_node = None
        self.up_node = None
        self.down_node = None
        self.h0 = h0
        self.h1 = h1
        self.w0 = w0
        self.w1 = w1

    def split_judge(self):
        img = self.img
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        area = img[h0: h1, w0: w1]
        var_value = np.var(area)  # 计算方差
        if var_value > 1.5:
            return True  # 需要分裂
        else:
            return False  # 不需要分裂

    def cal_mean(self):
        img = self.img
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        area = img[h0: h1, w0: w1]
        mean_value = np.mean(area)  # 计算均值
        # print('mean_value',mean_value)
        return mean_value

    def draw_contour_img(self, contour_img):
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        for h in range(h0-1, h1):
            for w in range(w0-1, w1):
                contour_img[h][w] = 255  # 填充为白色
        return contour_img

    def split_node(self):
        sub_node1 = ImgNode(self.img, self, self.h0, int(
            (self.h0+self.h1)/2), self.w0, int((self.w0+self.w1)/2))
        sub_node2 = ImgNode(self.img, self, self.h0, int(
            (self.h0+self.h1)/2), int((self.w0+self.w1)/2), self.w1)
        sub_node3 = ImgNode(self.img, self, int(
            (self.h0+self.h1)/2), self.h1, self.w0, int((self.w0+self.w1)/2))
        sub_node4 = ImgNode(self.img, self, int(
            (self.h0+self.h1)/2), self.h1, int((self.w0+self.w1)/2), self.w1)

        sub_node1.left_node = self.left_node
        sub_node1.right_node = sub_node2
        sub_node1.up_node = self.up_node
        sub_node1.down_node = sub_node3

        sub_node2.left_node = sub_node1
        sub_node2.right_node = self.right_node
        sub_node2.up_node = self.up_node
        sub_node2.down_node = sub_node4

        sub_node3.left_node = self.left_node
        sub_node3.right_node = sub_node4
        sub_node3.up_node = sub_node1
        sub_node3.down_node = self.down_node

        sub_node4.left_node = sub_node3
        sub_node4.right_node = self.right_node
        sub_node4.up_node = sub_node2
        sub_node4.down_node = self.down_node

        if self.left_node is not None:
            self.left_node.right_node = sub_node1
        if self.up_node is not None:
            self.up_node.down_node = sub_node2
        if self.down_node is not None:
            self.down_node.up_node = sub_node3
        if self.right_node is not None:
            self.right_node.left_node = sub_node4

        self.sub_node1 = sub_node1
        self.sub_node2 = sub_node2
        self.sub_node3 = sub_node3
        self.sub_node4 = sub_node4
        return sub_node1, sub_node2, sub_node3, sub_node4

    def draw_node(self, img):
        point_color = (255, 255, 255)
        thickness = 1
        lineType = 4
        cv2.rectangle(img, (self.w0, self.h0), (self.w1, self.h1),
                      point_color, thickness, lineType)
        return img


def split(node, draw_img):
    if node.h1-node.h0 >= 1 and node.w1-node.w0 >= 1:
        draw_img = node.draw_node(draw_img)
    if node.split_judge() and node.h1-node.h0 >= 2 and node.w1-node.w0 >= 2:
        node.split_node()
        draw_img = split(node.sub_node1, draw_img)
        draw_img = split(node.sub_node2, draw_img)
        draw_img = split(node.sub_node3, draw_img)
        draw_img = split(node.sub_node4, draw_img)
    return draw_img


def is_leaf(node):
    if node.sub_node1 is None and node.sub_node2 is None and node.sub_node3 is None and node.sub_node4 is None:
        return True
    else:
        return False


def is_leaf_father(node):
    if is_leaf(node) is False and is_leaf(node.sub_node1) and is_leaf(node.sub_node2) and is_leaf(node.sub_node3) and is_leaf(node.sub_node4):
        return True
    else:
        return False


def find_leaf_father(node):
    if is_leaf_father(node):
        return node
    elif is_leaf(node):
        return None
    else:
        res1 = find_leaf_father(node.sub_node1)
        if res1 is not None:
            return res1
        res2 = find_leaf_father(node.sub_node2)
        if res2 is not None:
            return res2
        res3 = find_leaf_father(node.sub_node3)
        if res3 is not None:
            return res3
        res4 = find_leaf_father(node.sub_node4)
        if res4 is not None:
            return res4


def merge(node, contour_img):
    global i
    print('i=', i)
    i += 1

    global Node_List
    # print(Node_List)
    if node in Node_List:
        print('return')
        return contour_img

    else:
        Node_List.append(node)

    cv2.imshow('test', contour_img)
    cv2.imwrite('test.png', contour_img)

    threshold = 0.0
    contour_img = node.draw_contour_img(contour_img)
    self_mean = node.cal_mean()
    if node.right_node is not None:
        if abs(self_mean - node.right_node.cal_mean()) <= threshold:
            print('right', self_mean - node.right_node.cal_mean())
            # if self.right_node.left_node is self:
            #     self.right_node.left_node = None
            contour_img = merge(node.right_node,contour_img)
    if node.down_node is not None:
        if abs(self_mean - node.down_node.cal_mean()) <= threshold:
            print('down', self_mean - node.down_node.cal_mean())
            # if self.down_node.up_node is self:
            #     self.down_node.up_node = None
            contour_img = merge(node.down_node,contour_img)
    if node.up_node is not None:
        if abs(self_mean - node.up_node.cal_mean()) <= threshold:
            print('up', self_mean - node.up_node.cal_mean())
            # if self.up_node.down_node is self:
            #     self.up_node.down_node = None
            contour_img = merge(node.up_node,contour_img)
    if node.left_node is not None:
        if abs(self_mean - node.left_node.cal_mean()) <= threshold:
            print('left', self_mean - node.left_node.cal_mean())

            # if self.left_node.right_node is self:
            #     self.left_node.right_node = None
            contour_img = merge(node.left_node,contour_img)

    return contour_img


if __name__ == '__main__':
    sys.setrecursionlimit(10000)  # 设置递归层数

    img = cv2.imread('zju_logo.png', 0)
    origin_img = img.copy()
    cv2.imshow('origin_img', origin_img)
    draw_img = img.copy()
    # print(img.shape[0],img.shape[1])

    start_node = ImgNode(img, None, 0, img.shape[0], 0, img.shape[1])
    draw_img = start_node.draw_node(origin_img)

    draw_img = split(start_node, draw_img)
    cv2.imshow('draw_img', draw_img)
    cv2.imwrite('draw_img.png', draw_img)

    leaf_father = find_leaf_father(start_node)
    contour_img = np.zeros((int(img.shape[0]), int(img.shape[1])))
    contour_img = merge(leaf_father.sub_node4, contour_img)

    # contour_img = leaf_father.sub_node1.draw_contour_img(contour_img)

    cv2.imshow('contour_img', contour_img)
    cv2.imwrite('contour_img.png', contour_img)

    cv2.waitKey(0)
