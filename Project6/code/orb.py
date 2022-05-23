# https://blog.csdn.net/qq_40344307/article/details/97534809

import cv2
import matplotlib.pyplot as plt

origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片

numfeatures1 = 100 # 设置特征点数量
orb = cv2.ORB_create(numfeatures1) # 创建ORB对象
kp = orb.detect(origin_img1, None) # 寻找关键点
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

numfeatures2 = 300 # 设置特征点数量
orb = cv2.ORB_create(numfeatures2) # 创建ORB对象
kp = orb.detect(origin_img1, None) # 寻找关键点
keypoint_img2 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

# 绘制结果
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('ORB with feature number: %d' % numfeatures1)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2)
plt.title('ORB with feature number: %d' % numfeatures2)
plt.axis('off')

plt.show()