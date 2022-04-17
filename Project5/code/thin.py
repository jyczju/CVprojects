import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Zhang Suen thining algorythm
def img_thin(img):
    # get shape
    h, w = img.shape

    # prepare out image
    out = np.zeros((h, w), dtype=int)
    out[img > 0] = 1
    out = 1-out

    # i = 0 # 调试用
    while True:
        delet_node1 = []
        delet_node2 = []

        # step 1 ( rasta scan )
        for x in range(1, h-1):
            for y in range(1, w-1):
                
                # condition 1
                if out[x, y] == 1:
                    # condition 2
                    num_of_one = np.sum(out[x-1:x+2, y-1:y+2])-1
                    if num_of_one >= 2 and num_of_one <= 6:
                        
                        # condition 3
                        if count_0_to_1(out, x, y) == 1:
                            # condition 4
                            if out[x-1, y]*out[x+1, y]*out[x, y+1] == 0 and out[x, y-1]*out[x+1, y]*out[x, y+1] == 0:
                                    delet_node1.append((x, y))

        for node in delet_node1:
            out[node] = 0

        # step 2 ( rasta scan )
        for x in range(1, h-1):
            for y in range(1, w-1):
                
                # condition 1
                if out[x, y] == 1:
                    # condition 2
                    num_of_one = np.sum(out[x-1:x+2, y-1:y+2])-1
                    if num_of_one >= 2 and num_of_one <= 6:
                        
                        # condition 3
                        if count_0_to_1(out, x, y) == 1:
                            # condition 4
                            if out[x-1, y]*out[x, y-1]*out[x+1, y] == 0 and out[x-1, y]*out[x, y-1]*out[x, y+1] == 0:
                                    delet_node2.append((x, y))
        
        for node in delet_node2:
            out[node] = 0
        
        # 调试用
        # tmp = out.copy()
        # tmp = 1-tmp
        # tmp = tmp.astype(np.uint8) * 255
        # cv2.imwrite('./results/'+str(i)+'.png', tmp)
        # i += 1

        # if not any pixel is changed
        if len(delet_node1) ==0 and len(delet_node2) == 0:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out

def count_0_to_1(img, x, y):
    num = 0
    if (img[x-1, y+1] - img[x-1, y]) == 1:
        num += 1
    if (img[x, y+1] - img[x-1, y+1]) == 1:
        num += 1
    if (img[x+1, y+1] - img[x, y+1]) == 1:
        num += 1
    if (img[x+1, y] - img[x+1,y+1]) == 1:
        num += 1
    if (img[x+1, y-1] - img[x+1, y]) == 1:
        num += 1
    if (img[x, y-1] - img[x+1, y-1]) == 1:
        num += 1
    if (img[x-1, y-1] - img[x, y-1]) == 1:
        num += 1
    if (img[x-1, y] - img[x-1, y-1]) == 1:
        num += 1
    return num

if "__main__" == __name__:
    img = cv2.imread("filter.png", 0)
    cv2.imshow("original", img)

    out = img_thin(img)

    cv2.imshow("result", out)
    cv2.waitKey(0)

