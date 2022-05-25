# https://blog.csdn.net/zhangziju/article/details/79754652
# https://blog.csdn.net/yukinoai/article/details/88912586

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
 
origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片
origin_img2 = cv2.imread('origin_img2.jpg') # 读取图片

gray1 = cv2.cvtColor(origin_img1 , cv2.COLOR_BGR2GRAY) # 转换为灰度图
gray2 = cv2.cvtColor(origin_img2 , cv2.COLOR_BGR2GRAY) # 转换为灰度图

start_time = time.time()
numfeatures = 200 # 特征点数量
sift = cv2.xfeatures2d.SIFT_create(numfeatures) # 创建SIFT对象
kp1, des1 = sift.detectAndCompute(gray1, None) # 计算SIFT特征点和描述符
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp1,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制特征点

kp2, des2 = sift.detectAndCompute(gray2, None) # 计算SIFT特征点和描述符
keypoint_img2 = cv2.drawKeypoints(origin_img2, kp2,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制特征点

end_time = time.time()
print('time:',end_time-start_time,'s')
print('feature number:',len(kp1))

# BFMatcher匹配
bf = cv2.BFMatcher() # 创建BFMatcher对象
matches = bf.knnMatch(des1,des2, k=2) # 使用knn进行计算匹配

# NNDR匹配
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance: # 最近点距离/次近点距离<0.84
        good.append([m])

match_img = cv2.drawMatchesKnn(origin_img1, kp1, origin_img2, kp2, good, None, flags=2) # 绘制匹配点

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