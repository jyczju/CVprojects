import numpy as np

def dilate(img, dilate_time=1):
    '''腐蚀'''
    h, w = img.shape
    # kernel
    kernal = np.array(((0, 1, 0),(1, 1, 1),(0, 1, 0)), dtype=int)
    # each dilate time
    out = img.copy()
    for i in range(dilate_time):
        tmp = np.pad(out, (1, 1), 'edge') # 边缘填充
        for x in range(1, h):
            for y in range(1, w):
                if np.sum(kernal * tmp[x-1:x+2, y-1:y+2]) >= 255:
                    out[x, y] = 255
    return out

def erode(img, erode_time=1):
    '''膨胀'''
    h, w = img.shape
    # kernel
    kernal = np.array(((0, 1, 0),(1, 1, 1),(0, 1, 0)), dtype=int)
    # each erode time
    out = img.copy()
    for i in range(erode_time):
        tmp = np.pad(out, (1, 1), 'edge') # 边缘填充
        for x in range(1, h):
            for y in range(1, w):
                if np.sum(kernal * tmp[x-1:x+2, y-1:y+2]) < 255*5:
                    out[x, y] = 0
    return out