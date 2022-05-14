import numpy as np
import cv2
import matplotlib.pyplot as plt
 
img = cv2.imread('test32.jpg', 0)
 
surf = cv2.xfeatures2d.SURF_create(5000)
 
kp = surf.detect(img, None)
 
img2 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# 不检查关键点的方向
# surf.setUpright(True)
# 修改阈值
surf.setHessianThreshold(10000)
kp = surf.detect(img, None)
img3 = cv2.drawKeypoints(img, kp, None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
plt.subplot(121), plt.imshow(img2),
plt.title('Dstination'), plt.axis('off')
plt.subplot(122), plt.imshow(img3),
plt.title('Dstination'), plt.axis('off')
plt.show()