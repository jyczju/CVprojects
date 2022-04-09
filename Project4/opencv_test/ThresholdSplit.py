import cv2
import matplotlib.pyplot as plt

# 1. 图像读取
img = cv2.imread('ex_red_3.jpg',0)

# 2.固定阈值
ret, th1 = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)


# 3.自适应阈值
# 3.1 邻域内求均值
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 27, 4) # 11 4
th2 = cv2.medianBlur(th2,5)
# 3.2 邻域内高斯加权
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 29, 4)
# 4 结果绘制
# titles = ['原图', '全局阈值(v = 127)', '自适应阈值（求均值）', '自适应阈值（高斯加权）']
titles = ['Origin', 'Threshold(v=40)', 'AdaptiveThreshold(mean)', 'AdaptiveThreshold(Guass)']
images = [img, th1, th2, th3]
plt.figure(figsize=(10,6))
for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=8)
    plt.xticks([]), plt.yticks([])
plt.show()