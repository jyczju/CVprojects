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

