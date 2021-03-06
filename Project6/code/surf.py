import cv2
import matplotlib.pyplot as plt
 
origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片

gray1 = cv2.cvtColor(origin_img1 , cv2.COLOR_BGR2GRAY) # 将图片转为灰度图

HessianThreshold1 = 5500 # 设置阈值
surf = cv2.xfeatures2d.SURF_create(HessianThreshold1) # 创建SURF对象
kp = surf.detect(gray1, None) # 寻找关键点
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

HessianThreshold2 = 2000 # 设置阈值
surf = cv2.xfeatures2d.SURF_create(HessianThreshold2) # 创建SURF对象
kp = surf.detect(gray1, None) # 寻找关键点
keypoint_img2 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

keypoint_img1 = cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB) # 将图片转为RGB图
keypoint_img2 = cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB) # 将图片转为RGB图

# 绘制结果
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('SURF with Hessian Threshold: %d' % HessianThreshold1)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2)
plt.title('SURF with Hessian Threshold: %d' % HessianThreshold2)
plt.axis('off')

plt.show()