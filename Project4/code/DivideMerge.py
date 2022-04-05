import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImgNode():
    def __init__(self,img,father_node,h0,h1,w0,w1):
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

    def divide_judge(self):
        img = self.img
        h0 = self.h0
        h1 = self.h1
        w0 = self.w0
        w1 = self.w1
        area = img[h0 : h1, w0 : w1]
        var = np.var(area) # 计算方差
        if var > 1.5:
            return True # 需要分裂
        else:
            return False # 不需要分裂

    def divide_node(self):
        sub_node1 = ImgNode(self.img, self, self.h0,int((self.h0+self.h1)/2),self.w0,int((self.w0+self.w1)/2))
        sub_node2 = ImgNode(self.img, self, self.h0,int((self.h0+self.h1)/2),int((self.w0+self.w1)/2),self.w1)
        sub_node3 = ImgNode(self.img, self, int((self.h0+self.h1)/2),self.h1,self.w0,int((self.w0+self.w1)/2))
        sub_node4 = ImgNode(self.img, self, int((self.h0+self.h1)/2),self.h1,int((self.w0+self.w1)/2),self.w1)
        
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

        self.sub_node1 = sub_node1
        self.sub_node2 = sub_node2
        self.sub_node3 = sub_node3
        self.sub_node4 = sub_node4
        return sub_node1,sub_node2,sub_node3,sub_node4
        
    def draw_node(self, img):
        point_color = (255, 255, 255)
        thickness = 1
        lineType = 4
        cv2.rectangle(img, (self.w0, self.h0), (self.w1, self.h1), point_color, thickness, lineType)
        return img


def divide(node,draw_img):
    if node.divide_judge() and node.h1-node.h0 >= 1 and node.w1-node.w0 >= 1:
        draw_img = node.draw_node(draw_img)
        node.divide_node()
        draw_img = divide(node.sub_node1,draw_img)
        draw_img = divide(node.sub_node2,draw_img)
        draw_img = divide(node.sub_node3,draw_img)
        draw_img = divide(node.sub_node4,draw_img)
    return draw_img



if __name__ == '__main__':
    img = cv2.imread('zju_logo.png',0)
    origin_img = img.copy()
    cv2.imshow('origin_img',origin_img)
    draw_img = img.copy()
    print(img.shape[0],img.shape[1])
    start_node = ImgNode(img,None,0,img.shape[0],0,img.shape[1])
    draw_img = start_node.draw_node(origin_img)

    draw_img = divide(start_node,draw_img)
    cv2.imshow('draw_img',draw_img)

    cv2.waitKey(0)
