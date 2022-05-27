

# 图像特征提取算法上机实践报告

<center>蒋颜丞，自动化(电气)1903，3190102563  </center>

### 1 实验内容和要求

​		编程实现（可以调用opencv库）SIFT、SURF、ORB等图像特征提取算法，并比较各算法性能。

### 2 实验原理

#### 2.1 SIFT

##### 2.1.1 SIFT特征的性质

（1）不变性：对图像的旋转和尺度变化具有不变性，对三维视角变化和光照变化具有很强的适应性，局部特征在遮挡和场景杂乱时仍保持不变性；
（2）辨别力强：特征之间相互区分的能力强，有利于匹配；
（3）数量多：一般 500 × 500 的图像能提取出约2000个特征点；
（4）扩展性强：可以很方便的与其他形式的特征向量进行联合。

##### 2.1.2 SIFT特征提取

（1）尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。

（2）关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。

（3）方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。

（4）生成描述子：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种描述子，这种描述子允许比较大的局部形状的变形和光照变化。

##### 2.1.3 SIFT特征匹配

​		当两幅图像的SIFT特征向量生成后，就可以采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取图1中的某个关键点，使用BBF (Best Bin First) 算法进行搜索，找到图2中的距离最近的两个关键点。采用NNDR (Nearest Neighbor Distance Ratio)作为匹配准则，若$NNDR=\frac{d(m,m_1)}{d(m,m_2)}=\frac{最近距离}{次近距离}$小于某个阈值，则认为图1中的该关键点与图2中的最近点为一对匹配点。



#### 2.2 SURF

##### 2.2.1 SURF特征提取

（1）构造Hessian矩阵，计算变换图：先求出图像中每个像素点的Hessian矩阵$\mathcal{H}(\mathrm{x}, \sigma)=\left[\begin{array}{ll}
L_{x x}(\mathrm{x}, \sigma) & L_{x y}(\mathrm{x}, \sigma) \\
L_{x y}(\mathrm{x}, \sigma) & L_{y y}(\mathrm{x}, \sigma)
\end{array}\right]$，然后计算变换图，该变换图的每个像素即为Hessian矩阵行列式的近似值$\operatorname{det}\left(H_{a p p r o x}\right)=\mathrm{L}_{x x} \mathrm{~L}_{y y}-\left(0.9 \mathrm{~L}_{x y}\right)^{2}$，其中Hessian矩阵的计算可以使用模板卷积来近似，同时引入积分图像来提高运算速度。

（2）构造高斯金字塔：对于SIFT算法，每一组图像的大小是不一样的，下一组是上一组图像的降采样，在每一组中的几幅图像，他们的大小相同，但采用的尺度σ不同，且在模糊的过程中，高斯模板大小不变，只是尺度σ改变。对于SURF算法，图像的大小总是不变的，改变的只是高斯模糊模板的尺寸。

（3）定位特征点：首先设置阈值，确定候选兴趣点，然后，将经过Hessian矩阵处理过的每个像素点与其三维邻域的26个点进行大小比较，如果它是这26个点中的最大值或者最小值，则保留，最后，采用拟合3D二次曲线内插子像素进行精确定位。

（4）确定特征点主方向：以特征点为中心，计算半径为6σ的邻域内，统计60度扇形内所有点在水平和垂直方向的Haar小波响应总和，并给这些响应值赋高斯权重系数，使得靠近特征点的响应贡献大，而远离特征点的响应贡献小，然后60度范围内的响应相加以形成新的矢量，遍历整个圆形区域，选择最长矢量的方向为该特征点的主方向。这样，通过特征点逐个进行计算，得到每一个特征点的主方向。

（5）构造特征描述子：在特征点周围取一个正方形框，框的边长为20σ。然后把该框分为16个子区域，每个子区域统计25个像素的x方向和y方向的haar小波特征。该haar小波特征为x方向值之和，x方向绝对值之和，y方向之和，y方向绝对值之和。

##### 2.2.2 SURF特征匹配

（1）在检测特征点的过程中，计算了Hessian矩阵的行列式，与此同时，计算得到了Hessian矩阵的迹，矩阵的迹为对角元素之和。按照亮度的不同，可以将特征点分为两种，第一种为特征点及其周围小邻域的亮度比背景区域要亮，Hessian矩阵的迹为正；另外一种为特征点及其周围小邻域的亮度比背景区域要暗，Hessian矩阵为负值。根据这个特性，首先对两个特征点的Hessian的迹进行比较。如果同号，说明两个特征点具有相同的对比度；如果是异号的话，说明两个特征点的对比度不同，放弃特征点之间后续的相似性度量。

（2）对于两个特征点描述子的相似性度量，我们采用欧式距离进行计算：
$$
\operatorname{Dis}_{i j}=\left[\sum_{k=0}^{k=n}\left(X_{i k}-X_{j k}\right)^{2}\right]^{1 / 2}
$$
​		对于待配准图上的特征点，计算它到参考图像上所有特征点的欧氏距离，得到一个距离集合。通过对距离集合进行比较运算得到小欧氏距离和次最小欧式距离。设定一个阈值，当最小欧氏距离和次最小欧式距离的比值小于该阈值时，认为特征点与对应最小欧氏距离的特征点是匹配的，否则没有点与该特征点相匹配。阈值越小，匹配越稳定，但极值点越少，



#### 2.3 ORB

​		ORB(Oriented FAST and Rotated BRIEF)算法结合了FAST与BRIEF算法，并给FAST特征点增加了方向性，使得特征点具有旋转不变性，并提出了构造金字塔方法，解决尺度不变性。其中，特征提取是由FAST(Features from Accelerated Segment Test)算法发展而来的，特征描述是根据BRIEF(Binary Robust Independent Elementary Features)算法改进的。ORB特征是将FAST特征点的检测方法与BRIEF特征描述子结合起来，并在它们原来的基础上做了改进与优化。ORB主要解决BRIEF描述子不具备旋转不变性的问题。


### 3 源代码

sift_match.py

```python
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
```

surf_match.py

```python
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
plt.subplot(121)
plt.imshow(keypoint_img1)
plt.title('origin_img1 (Hessian Threshold: %d)' % HessianThreshold)
plt.axis('off')

plt.subplot(122)
plt.imshow(keypoint_img2)
plt.title('origin_img2 (Hessian Threshold: %d)' % HessianThreshold)
plt.axis('off')
plt.show()

cv2.imshow('match_img', match_img)
cv2.waitKey(0)
```

orb_match.py

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

origin_img1 = cv2.imread('origin_img1.jpg') # 读取图片
origin_img2 = cv2.imread('origin_img2.jpg') # 读取图片

start_time = time.time()
numfeatures = 200 # 特征点数量
orb = cv2.ORB_create(numfeatures) # 创建ORB对象
kp1, des1 = orb.detectAndCompute(origin_img1, None) # 寻找关键点
keypoint_img1 = cv2.drawKeypoints(origin_img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # 绘制关键点

kp2, des2 = orb.detectAndCompute(origin_img2, None) # 寻找关键点
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
    if m.distance < 0.84*n.distance: # 最近点距离/次近点距离<0.84
        good.append([m])

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
```



### 4 实验结果与分析

[(26条消息) 应用OpenCV和Python进行SIFT算法的实现_章子雎Kevin的博客-CSDN博客_python sift](https://blog.csdn.net/zhangziju/article/details/79754652)



对比分析



| 特征提取方法 |                         特征提取结果                         | 特征提取所用时间 |                           特征匹配                           |
| :----------: | :----------------------------------------------------------: | :--------------: | :----------------------------------------------------------: |
|     SIFT     | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\sift_detect-16535561274031.png" alt="sift_detect" style="zoom:70%;" /> |      1.159s      | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\sift_match.png" alt="sift_match" style="zoom:70%;" /> |
|     SURF     | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\surf_detect.png" alt="surf_detect" style="zoom:70%;" /> |      0.608s      | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\surf_match.png" alt="surf_match" style="zoom:70%;" /> |
|     ORB      | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\orb_detect.png" alt="orb_detect" style="zoom:70%;" /> |      0.068s      | <img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\orb_match.png" alt="orb_match" style="zoom:70%;" /> |







综上所述，可知SURF采用Henssian矩阵获取图像局部最值还是十分稳定的，但是在求主方向阶段太过于依赖局部区域像素的梯度方向，有可能使得找到的主方向不准确，后面的特征向量提取以及匹配都严重依赖于主方向，即使不大偏差角度也可以造成后面特征匹配的放大误差，从而匹配不成功；另外图像金字塔的层取得不足够紧密也会使得尺度有误差，后面的特征向量提取同样依赖相应的尺度，在这个问题上我们只能采用折中解决方法：取适量的层然后进行插值



总体来说ORB在上述实验实例中的效果还是很好的。我们已经讲完SIFT，SURF还有ORB，我们来对比一下这三个算法。
 计算速度：            ORB>>SURF>>SIFT（各差一个量级）
 旋转鲁棒性：        SURF>ORB~SIFT（～表示差不多）
 模糊鲁棒性：        SURF>ORB~SIFT
 尺度变换鲁棒性： SURF>SIFT>ORB（ORB尺度变换性很弱）
 在日常应用中，有SURF基本就不用考虑SIFT，SURF基本就是SIFT的全面升级版，当然也有其他SIFT的改进版比如Affine SIFT的效果就要比SUFR要好更多，但是计算时间也有延长，而ORB的强点在于计算时间。ORB主要还是在VSLAM中应用较多，场景变化不明显，但是需要高速的计算时间，这正好符合ORB。





效果：

特征提取数量、质量

匹配效果



开销：

运行时间







<img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project6\report\image-20220525220915562.png" alt="image-20220525220915562" style="zoom:80%;" />

SURF 速度快，较 SIFT 速度检测和匹配，有3倍速度提高，性能和 SIFT 大体相当

SURF 在图像模糊和旋转不变上优于 SIFT

SURF 在图像视点变化和光照变化上略差于 SIFT
