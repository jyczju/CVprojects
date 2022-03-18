'''
直方图均衡化，浙江大学控制学院《数字图像处理与机器视觉》第二次作业
jyczju
2022/3/9 v1.0
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Img_Hist(img):
    '''
    计算图像的直方图
    输入：灰度图像
    输出：以一维向量保存的直方图数据
    '''
    hist=np.zeros(256,dtype=int) # 初始化直方图数据

    # 统计直方图数据
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            hist[k] += 1
    
    return hist

def Equal_Hist(hist, img):
    '''
    直方图均衡处理
    输入：原始图像直方图数据，原始图像
    输出：处理后图像
    '''
    Tr = Cal_Tr(hist, img) # 计算Tr

    # plt.figure(0)
    # plt.plot(np.arange(256), Tr, 'r', linewidth=1.5, c='black')
    # plt.tick_params(labelsize=15)
    # plt.title("Tr",fontdict={'weight':'normal','size': 20})
    # plt.xlabel("Histogram",fontdict={'weight':'normal','size': 15})
    # plt.ylabel("Tr",fontdict={'weight':'normal','size': 15})
    
    new_img = np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.uint8)
    
    # 对原图像进行映射
    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = Tr[img[k][l]]
            
    return new_img

def Cal_Tr(hist, img):
    '''
    计算Tr
    输入：原始图像直方图数据，原始图像
    输出：Tr一维数组
    '''
    Pr=np.zeros(256,dtype=float)  # 初始化频率分布
    N = img.shape[0]*img.shape[1]
    for i in range(256):
        Pr[i]=hist[i]/N # 统计频率

    # 计算Tr
    Tr=np.zeros(256,dtype=float)  # 初始化Tr
    temp = 0
    for m in range(256):
        temp += 255*Pr[m]
        Tr[m] = np.uint8(round(temp))

    return Tr


if __name__ == '__main__':

    # 读取原始图像
    img = cv2.imread('GrayPhoto.jpg', cv2.IMREAD_GRAYSCALE)

    #计算原图的直方图
    origin_hist = Img_Hist(img)

    # 直方图均衡化
    new_img = Equal_Hist(origin_hist,img)

    # 计算处理后图像的直方图
    new_hist = Img_Hist(new_img)

    # cv2.imshow('origin_image',img)
    # cv2.imshow('new_image',new_img)
    cv2.imwrite('new_image.jpg', new_img)

    plt.figure(1)

    plt.subplot(121),plt.imshow(img, cmap=plt.cm.gray),plt.title('origin_image',fontdict={'weight':'normal','size': 20}), plt.axis('off') #坐标轴关闭
    plt.subplot(122),plt.imshow(new_img, cmap=plt.cm.gray),plt.title('new_image',fontdict={'weight':'normal','size': 20}), plt.axis('off') #坐标轴关闭

    # 绘制灰度直方图
    plt.figure(2)
    plt.subplot( 1, 2, 1 )
    plt.plot(np.arange(256), origin_hist, 'r', linewidth=1.5, c='black')
    plt.tick_params(labelsize=15)
    plt.title("Origin",fontdict={'weight':'normal','size': 20})
    plt.xlabel("Histogram",fontdict={'weight':'normal','size': 15})
    plt.ylabel("Number of Pixels",fontdict={'weight':'normal','size': 15})

    plt.subplot( 1, 2, 2 )
    plt.plot(np.arange(256), new_hist, 'r', linewidth=1.5, c='black')
    plt.tick_params(labelsize=15)
    plt.title("New",fontdict={'weight':'normal','size': 20})
    plt.xlabel("Histogram",fontdict={'weight':'normal','size': 15})
    plt.ylabel("Number of Pixels",fontdict={'weight':'normal','size': 15})

    plt.show()

    print('Equalization has done.')

    K = cv2.waitKey(0)



