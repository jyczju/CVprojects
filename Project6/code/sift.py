import cv2
from matplotlib import pyplot as plt
 
origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片

gray1 = cv2.cvtColor(origin_img1 , cv2.COLOR_BGR2GRAY) # 将图片转为灰度图

numfeatures1 = 300 # 设置特征点数量
sift = cv2.xfeatures2d.SIFT_create(numfeatures1) # 创建SIFT对象
kp = sift.detect(gray1, None) # 寻找关键点
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

numfeatures2 = 6000 # 设置特征点数量
sift = cv2.xfeatures2d.SIFT_create(numfeatures2) # 创建SIFT对象
kp = sift.detect(gray1, None) # 寻找关键点
keypoint_img2 = cv2.drawKeypoints(origin_img1, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

keypoint_img1 = cv2.cvtColor(keypoint_img1, cv2.COLOR_BGR2RGB) # 将图片转为RGB图
keypoint_img2 = cv2.cvtColor(keypoint_img2, cv2.COLOR_BGR2RGB) # 将图片转为RGB图

# 绘制结果
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('SIFT with feature number: %d' % numfeatures1)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2)
plt.title('SIFT with feature number: %d' % numfeatures2)
plt.axis('off')

plt.show()