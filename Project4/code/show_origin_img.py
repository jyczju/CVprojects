import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    zju_logo = cv2.imread('zju_logo.png', 0) # 读入图像
    zjui_logo = cv2.imread('zjui_logo.png', 0) # 读入图像
    zju_logo_uneven = cv2.imread('zju_logo_uneven.png', 0) # 读入图像
    zjui_logo_uneven = cv2.imread('zjui_logo_uneven.png', 0) # 读入图像
    zju_logo_gauss = cv2.imread('zju_logo_gauss.png', 0) # 读入图像
    zjui_logo_gauss = cv2.imread('zjui_logo_gauss.png', 0) # 读入图像

    plt.figure(figsize=(9, 9.5))
    title_size = 12
    plt.subplot(321)
    plt.axis('off')
    plt.imshow(zju_logo,cmap='gray')
    plt.title("Figure 1: zju_logo",fontdict={'weight':'normal','size': title_size})

    plt.subplot(322)
    plt.axis('off')
    plt.imshow(zjui_logo,cmap='gray')
    plt.title("Figure 2: zjui_logo",fontdict={'weight':'normal','size': title_size})

    plt.subplot(323)
    plt.axis('off')
    plt.imshow(zju_logo_uneven,cmap='gray')
    plt.title("Figure 3: zju_logo_uneven",fontdict={'weight':'normal','size': title_size})

    plt.subplot(324)
    plt.axis('off')
    plt.imshow(zjui_logo_uneven,cmap='gray')
    plt.title("Figure 4: zjui_logo_uneven",fontdict={'weight':'normal','size': title_size})

    plt.subplot(325)
    plt.axis('off')
    plt.imshow(zju_logo_gauss,cmap='gray')
    plt.title("Figure 5: zju_logo_gauss",fontdict={'weight':'normal','size': title_size})

    plt.subplot(326)
    plt.axis('off')
    plt.imshow(zjui_logo_gauss,cmap='gray')
    plt.title("Figure 6: zjui_logo_gauss",fontdict={'weight':'normal','size': title_size})

    plt.show()

    cv2.waitKey(0)