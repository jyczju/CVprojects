'''
图像节点类，导入方法：from imgnode import ImgNode
浙江大学控制学院《数字图像处理与机器视觉》第三次作业
jyczju
2022/4/6 v1.0
'''
import cv2
import numpy as np


class ImgNode():
    '''
    图像节点类，用于构成四叉树
    '''
    Visited_List = []  # 已访问的节点，在merge方法中维护

    def __init__(self, img, father_node, h0, h1, w0, w1):
        self.img = img  # 原始图像
        self.father_node = father_node  # 父节点
        self.sub_node1 = None
        self.sub_node2 = None
        self.sub_node3 = None
        self.sub_node4 = None  # 子节点
        self.left_node = []  # 左节点
        self.right_node = []  # 右节点
        self.up_node = []  # 上节点
        self.down_node = []  # 下节点
        self.h0 = h0
        self.h1 = h1  # 当前节点在img上的h范围
        self.w0 = w0
        self.w1 = w1  # 当前节点在img上的w范围
        self.isleaf = True  # 用于存储当前节点是否为叶子节点

    def split_judge(self):
        '''
        判断当前节点是否需要分裂
        输入：当前节点
        输出：是否需要分裂
        '''
        var_value = self.cal_var()  # 计算当前节点的灰度方差
        if var_value > 3.6:  # 判断标准
            return True  # 需要分裂
        else:
            return False  # 不需要分裂

    def cal_var(self):
        '''
        计算当前节点的灰度方差
        输入：当前节点
        输出：方差
        '''
        img = self.img
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        area = img[h0: h1, w0: w1]
        var_value = np.var(area)  # 计算方差
        return var_value

    def cal_mean(self):
        '''
        计算当前节点的灰度均值
        输入：当前节点
        输出：均值
        '''
        img = self.img
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        area = img[h0: h1, w0: w1]
        mean_value = np.mean(area)  # 计算均值
        return mean_value

    def draw_region_img(self, region_img):
        '''
        绘制当前节点的区域图像
        输入：当前节点，区域图像
        输出：区域图像
        '''
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        # for h in range(h0-1, h1):
        #     for w in range(w0-1, w1): # 遍历当前节点范围内的所有像素
        #         region_img[h][w] = 255  # 填充为白色
        region_img[h0:h1, w0:w1] = 255
        return region_img

    def node_split(self):
        '''
        对当前节点进行分裂
        子节点说明：
         1 | 2 
        ———————
         3 | 4
        '''
        self.isleaf = False  # 当前节点不再是叶子节点
        sub_node1 = ImgNode(self.img, self, self.h0, int(
            (self.h0+self.h1)/2), self.w0, int((self.w0+self.w1)/2))  # 创建子节点1
        sub_node2 = ImgNode(self.img, self, self.h0, int(
            (self.h0+self.h1)/2), int((self.w0+self.w1)/2), self.w1)  # 创建子节点2
        sub_node3 = ImgNode(self.img, self, int(
            (self.h0+self.h1)/2), self.h1, self.w0, int((self.w0+self.w1)/2))  # 创建子节点3
        sub_node4 = ImgNode(self.img, self, int(
            (self.h0+self.h1)/2), self.h1, int((self.w0+self.w1)/2), self.w1)  # 创建子节点4

        # 链接各个子节点的上下左右节点
        sub_node1.left_node.extend(self.left_node)
        sub_node1.right_node.append(sub_node2)
        sub_node1.up_node.extend(self.up_node)
        sub_node1.down_node.append(sub_node3)

        sub_node2.left_node.append(sub_node1)
        sub_node2.right_node.extend(self.right_node)
        sub_node2.up_node.extend(self.up_node)
        sub_node2.down_node.append(sub_node4)

        sub_node3.left_node.extend(self.left_node)
        sub_node3.right_node.append(sub_node4)
        sub_node3.up_node.append(sub_node1)
        sub_node3.down_node.extend(self.down_node)

        sub_node4.left_node.append(sub_node3)
        sub_node4.right_node.extend(self.right_node)
        sub_node4.up_node.append(sub_node2)
        sub_node4.down_node.extend(self.down_node)

        # 链接当前节点的左节点的右节点
        for ln in self.left_node:
            if self in ln.right_node:
                ln.right_node.remove(self)
                if ln.h0 < sub_node1.h1 and ln.h1 > sub_node1.h0:
                    ln.right_node.append(sub_node1)
                if ln.h0 < sub_node3.h1 and ln.h1 > sub_node3.h0:
                    ln.right_node.append(sub_node3)

        # 链接当前节点的上节点的下节点
        for un in self.up_node:
            if self in un.down_node:
                un.down_node.remove(self)
                if un.w0 < sub_node1.w1 and un.w1 > sub_node1.w0:
                    un.down_node.append(sub_node1)
                if un.w0 < sub_node2.w1 and un.w1 > sub_node2.w0:
                    un.down_node.append(sub_node2)

        # 链接当前节点的下节点的上节点
        for dn in self.down_node:
            if self in dn.up_node:
                dn.up_node.remove(self)
                if dn.w0 < sub_node3.w1 and dn.w1 > sub_node1.w0:
                    dn.up_node.append(sub_node3)
                if dn.w0 < sub_node4.w1 and dn.w1 > sub_node4.w0:
                    dn.up_node.append(sub_node4)

        # 链接当前节点的右节点的左节点
        for rn in self.right_node:
            if self in rn.left_node:
                rn.left_node.remove(self)
                if rn.h0 < sub_node2.h1 and rn.h1 > sub_node2.h0:
                    rn.left_node.append(sub_node2)
                if rn.h0 < sub_node4.h1 and rn.h1 > sub_node4.h0:
                    rn.left_node.append(sub_node4)

        # 链接当前节点与各个子节点
        self.sub_node1 = sub_node1
        self.sub_node2 = sub_node2
        self.sub_node3 = sub_node3
        self.sub_node4 = sub_node4

    def is_leaf_father(self):
        '''
        判断当前节点是否是叶子节点的父节点
        '''
        if self.isleaf is False and self.sub_node1.isleaf and self.sub_node2.isleaf and self.sub_node3.isleaf and self.sub_node4.isleaf:
            return True
        else:
            return False

    def find_leaf_father(self):
        '''
        寻找一个叶子节点的父节点
        输入：一个起始节点（不能是叶子节点）
        输出：叶子节点的父节点
        '''
        if self.is_leaf_father():
            return self
        elif self.isleaf:
            return None
        else:  # 从子节点出发递归调用
            res1 = self.sub_node1.find_leaf_father()
            if res1 is not None:
                return res1
            res2 = self.sub_node2.find_leaf_father()
            if res2 is not None:
                return res2
            res3 = self.sub_node3.find_leaf_father()
            if res3 is not None:
                return res3
            res4 = self.sub_node4.find_leaf_father()
            if res4 is not None:
                return res4

    def draw_node(self, img):
        '''
        在img中绘制当前节点
        输入：当前节点，欲绘制的图像
        输出：绘制后的图像
        '''
        point_color = (255, 255, 255)  # 颜色
        thickness = 1  # 粗细
        lineType = 4  # 线型
        cv2.rectangle(img, (self.w0, self.h0), (self.w1, self.h1),
                      point_color, thickness, lineType)  # 绘制矩形
        return img

    def split(self, draw_img, min_area=(1, 1)):
        '''
        区域分裂
        输入：欲绘制的图像，最小区域大小
        输出：绘制好的图像
        '''
        if self.split_judge() and self.h1-self.h0 >= 2*min_area[0] and self.w1-self.w0 >= 2*min_area[1]:  # 符合分裂条件
            self.node_split()  # 分裂当前节点
            # 递归分裂当前节点的子节点
            draw_img = self.sub_node1.split(draw_img, min_area)
            draw_img = self.sub_node2.split(draw_img, min_area)
            draw_img = self.sub_node3.split(draw_img, min_area)
            draw_img = self.sub_node4.split(draw_img, min_area)

        if self.h1-self.h0 >= min_area[0] and self.w1-self.w0 >= min_area[1]:
            draw_img = self.draw_node(draw_img)  # 绘制当前节点

        return draw_img

    def merge(self, region_img, threshold=5.0):
        '''
        区域合并
        输入：当前节点，欲绘制的区域二值图像，相似性判断阈值（若两个区域的灰度均值之差小于threshold，则认为这两块区域可以合并）
        输出：绘制的区域二值图像
        '''
        if self in ImgNode.Visited_List:
            return region_img  # 如果当前节点已被访问过，则不再访问
        else:
            ImgNode.Visited_List.append(self)  # 将当前节点加入访问表

        region_img = self.draw_region_img(region_img)  # 绘制区域二值图像
        self_mean = self.cal_mean()  # 计算当前节点灰度均值

        for rn in self.right_node:  # 遍历所有右侧节点
            # 右侧节点与当前节点类似，且未被访问过
            if abs(self_mean - rn.cal_mean()) <= threshold and rn not in ImgNode.Visited_List:
                # print('right', self_mean - rn.cal_mean())
                region_img = rn.merge(region_img, 5)  # 对右侧节点进行递归合并

        for dn in self.down_node:  # 遍历所有下方节点
            # 下方节点与当前节点类似，且未被访问过
            if abs(self_mean - dn.cal_mean()) <= threshold and dn not in ImgNode.Visited_List:
                # print('down', self_mean - dn.cal_mean())
                region_img = dn.merge(region_img, 5)  # 对下方节点进行递归合并

        for un in self.up_node:  # 遍历所有上方节点
            # 上方节点与当前节点类似，且未被访问过
            if abs(self_mean - un.cal_mean()) <= threshold and un not in ImgNode.Visited_List:
                # print('up', self_mean - un.cal_mean())
                region_img = un.merge(region_img, 5)  # 对上方节点进行递归合并

        for ln in self.left_node:  # 遍历所有左侧节点
            # 左侧节点与当前节点类似，且未被访问过
            if abs(self_mean - ln.cal_mean()) <= threshold and ln not in ImgNode.Visited_List:
                # print('left', self_mean - ln.cal_mean())
                region_img = ln.merge(region_img, 5)  # 对左侧节点进行递归合并

        return region_img
