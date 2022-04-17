import numpy as np

def otsu(img_gray, GrayScale):
    u1=0.0 # 背景像素的平均灰度值
    u2=0.0 # 前景像素的平均灰度值
    th=0.0 # 阈值
    
    pixel_sum=img_gray.shape[0]*img_gray.shape[1] # 总的像素数目
    hist=np.zeros(GrayScale) # 各个灰度值的像素数目
    
    w=np.zeros(GrayScale)
    u=np.zeros(GrayScale)
    
    # 统计各个灰度值的像素个数
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            #默认灰度图像的像素值范围为GrayScale
            PixGray = img_gray[i][j]
            hist[PixGray] += 1

    # 计算w(k)
    for i in range(GrayScale):
        w[i]=np.sum(hist[:i])*1.0/pixel_sum

    # 计算u(k)
    for i in range(GrayScale):
        sum_temp = 0
        for j in range(i+1):
            sum_temp += hist[j]*j
        u[i]=sum_temp*1.0/pixel_sum
    
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for thi in range(1,GrayScale):
        
        w1=w[thi]-w[0] # 前景像素的比例
        w2=w[-1]-w[thi] # 背景像素的比例

        if w1!=0 and w2!=0:
            # 前景像素的平均灰度值
            u1 = (u[thi]-u[0]) * 1.0 / w1
            # 背景像素的平均灰度值
            u2 = (u[-1]-u[thi])* 1.0 / w2

            tem_var=w1*np.power((u1-u[-1]),2)+w2*np.power((u2-u[-1]),2) # 计算类间方差公式
            #判断当前类间方差是否为最大值。
            if Max_var<tem_var:
                Max_var=tem_var # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th=thi
    return th 

def threshold(img, th):
    th_img = np.zeros(img.shape, dtype=np.uint8)
    th_img[img>=th] = 255
    th_img[img<th] = 0
    return th_img

