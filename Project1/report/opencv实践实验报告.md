# openCV实践报告

<center>蒋颜丞，自动化(电气)1903，3190102563  </center>




### 1 实验要求

下载并安装opencCV库，实践图像和视频显示（自选一幅图像和视频，打上个人信息，并显示）。



### 2 实验内容

#### 2.1 图像

##### 2.1.1 实验过程

用imread方法以RGB三通道的格式打开jpg格式图像，使用openCV自带的putText方法可在图片上添加英文。为了在图像上添加中文，需要使用到PIL中的ImageDraw包，利用其中的text方法向图片上添加中文。为了方便使用，我将添加中文的代码封装成了cv2AddChineseText函数，使用时只需调用该函数即可。



##### 2.1.2 源代码

```python
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def cv2AddChineseText(img, text, position, textColor=(0, 0, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("./fonts/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

img = cv2.imread('DSC04229.jpg', cv2.IMREAD_COLOR)
img_text=img.copy()

# cv2.putText(img_text, 'Name:Jiang Yancheng', (10, 110),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, 0)
# cv2.putText(img_text, 'ID number:3190102563', (10, 150),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA, 0)
img_text = cv2AddChineseText(img_text, "姓名:蒋颜丞" ,  (10, 10), textColor=(0, 0, 0), textSize=40)
img_text = cv2AddChineseText(img_text, "学号:3190102563" ,  (10, 60), textColor=(0, 0, 0), textSize=40)

cv2.imshow('image', img_text)
cv2.imwrite('img_text.jpg', img_text)
K = cv2.waitKey(0)
```



##### 2.1.3 实验结果

<img src="D:\浙江大学\课程\2022春夏课程\数字图像处理与机器视觉\CVprojects\Project1\report\img_text.jpg" alt="img_text" style="zoom:67%;" />



#### 2.2 视频

##### 2.2.1 实验过程

逐帧处理，逐帧保存



##### 2.2.2 源代码

```python

```



##### 2.2.3 实验结果

(放一张截图)



