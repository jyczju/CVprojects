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

cap=cv2.VideoCapture("video.mp4")

# 循环读取图片
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2AddChineseText(frame, "姓名:蒋颜丞" ,  (10, 90), textColor=(255, 255, 255), textSize=40)
        frame = cv2AddChineseText(frame, "学号:3190102563" ,  (10, 140), textColor=(255, 255, 255), textSize=40)
        # frame = cv2AddChineseText(frame, "姓名:蒋颜丞" ,  (10, 110), textColor=(255, 255, 255), textSize=40)
        # frame = cv2AddChineseText(frame, "学号:3190102563" ,  (10, 160), textColor=(255, 255, 255), textSize=40)
        cv2.imshow("video", frame)

    else:
        print("视频播放完成！")
        break

    # 退出播放
    key = cv2.waitKey(25)
    if key == 27:  # 按键esc
        break

cap.release()
cv2.destroyAllWindows()
