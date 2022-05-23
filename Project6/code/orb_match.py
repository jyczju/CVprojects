# https://blog.csdn.net/qq_40344307/article/details/97534809

import cv2
import numpy as np
import matplotlib.pyplot as plt

origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片
origin_img2 = cv2.imread('origin_img2.jpg') # 读取图片

numfeatures = 300 # 特征点数量
orb = cv2.ORB_create(numfeatures) # 创建ORB对象
kp1, des1 = orb.detectAndCompute(origin_img1, None) # 寻找关键点
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

kp2, des2 = orb.detectAndCompute(origin_img2, None) # 寻找关键点
keypoint_img2 = cv2.drawKeypoints(origin_img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

bf = cv2.BFMatcher() # 创建BFMatcher对象
matches = bf.knnMatch(des1,des2,k=1) # 使用knn进行计算匹配
# 计算最大距离和最小距离
m0 = matches[0][0]
min_distance = m0.distance
for x in matches:
    if x[0].distance < min_distance:
        min_distance = x[0].distance  

# print("min_distance:",min_distance)
good = [] # 创建空列表
for x in matches:
    if x[0].distance <= 1.85 * min_distance: # 当描述符之间的距离大于1.85*最小距离时，认为匹配有误
        good.append(x) # 将通过筛选的匹配点添加到good列表

# # BFMatcher匹配
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # 调整ratio
# good = []
# for m,n in matches:
#     if m.distance < 0.9*n.distance: # 最近点距离小于0.75倍次近点距离
#         good.append([m])


match_img = cv2.drawMatchesKnn(origin_img1,kp1,origin_img2,kp2,good,None,flags=2) # 绘制匹配点

keypoint_img1 = cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB) # 转换为RGB图
keypoint_img2 = cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB) # 转换为RGB图

# 绘制结果
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('origin_img1 (feature number: %d)' % numfeatures)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2),
plt.title('origin_img2 (feature number: %d)' % numfeatures)
plt.axis('off')
plt.show()

cv2.imshow('match_img', match_img)
cv2.waitKey(0)