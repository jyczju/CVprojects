# https://blog.csdn.net/zhangziju/article/details/79754652
# https://blog.csdn.net/welcome_yu/article/details/105447393

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
 
origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片
origin_img2 = cv2.imread('origin_img2.jpg') # 读取图片

gray1 = cv2.cvtColor(origin_img1 , cv2.COLOR_BGR2GRAY) # 转换为灰度图
gray2 = cv2.cvtColor(origin_img2 , cv2.COLOR_BGR2GRAY) # 转换为灰度图

start_time = time.time()
HessianThreshold = 3400 # 设置阈值
surf = cv2.xfeatures2d.SURF_create(HessianThreshold) # 创建SURF对象
kp1, des1 = surf.detectAndCompute(gray1, None) # 计算SURF特征点和描述符
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

kp2, des2 = surf.detectAndCompute(gray2, None) # 计算SURF特征点和描述符
keypoint_img2 = cv2.drawKeypoints(origin_img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

end_time = time.time()
print('time:',end_time-start_time,'s')
print('feature number:',len(kp1))

# BFMatcher匹配
bf = cv2.BFMatcher() # 创建BFMatcher对象
matches = bf.knnMatch(des1,des2, k=2) # 使用knn进行计算匹配

# NNDR匹配
good = []
for m,n in matches:
    if m.distance < 0.85*n.distance: # 最近点距离/次近点距离<0.85
        good.append([m])

match_img = cv2.drawMatchesKnn(origin_img1,kp1,origin_img2,kp2,good,None,flags=2) # 绘制匹配点

keypoint_img1 = cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB) # 转换为RGB图
keypoint_img2 = cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB) # 转换为RGB图

# 绘制结果
plt.figure(figsize=(9,7))
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('origin_img1 (Hessian Threshold: %d)' % HessianThreshold)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2)
plt.title('origin_img2 (Hessian Threshold: %d)' % HessianThreshold)
plt.axis('off')
plt.savefig('surf_detect.png')
plt.show()

cv2.imshow('match_img', match_img)
cv2.imwrite('surf_match.png', match_img)
cv2.waitKey(0)