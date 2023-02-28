import math
from typing import List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray



def bilinearUpsampling(colorChannel: ndarray, scale: int,shape:Tuple[int,int,int]) -> ndarray:
    """
    :param colorChannel: a 2D ndarray representing a color channel
    :param scale: the scale factor
    :return: a 2D ndarray representing the upsampled color channel
    """
    result = np.full((shape[0], shape[1]), -1)
    for i in range(len(colorChannel)):
        for j in range(len(colorChannel[0])):
            if i*scale < len(result) and j*scale < len(result[0]):
                result[i * scale][j * scale] = colorChannel[i][j]
    print(result)

    for i in range(0, len(result)):
        if result[i][0] == -1:
            continue

        remainder = len(result[0]) % scale
        if remainder == 0:
            leftover = scale-1
        else:
            leftover = remainder-1

        for j in range(0, len(result[0])-leftover):
            if result[i][j] == -1:
                position_mod = j % scale
                result[i][j] = (result[i][j - position_mod] + result[i][j+(scale-position_mod)]) / 2

        for k in range(leftover):
            result[i][len(result[0])-k-1] = result[i][len(result[0])-1-leftover]


    print('====================')

    remainder = len(result) % scale
    if remainder == 0:
        leftover = scale - 1
    else:
        leftover = remainder - 1

    for i in range(0, len(result)-leftover):
        if result[i][0] != -1:
            continue

        position_mod = i % scale


        for j in range(0, len(result[0])):

                result[i][j] = (result[i-position_mod][j] + result[i+scale-position_mod][j]) / 2

    for k in range(leftover):
        for j in range(0, len(result[0])):
            result[len(result)-1-k][j] = result[len(result)-1-leftover][j]

    return result


def mergeColorChannel(yChanel: ndarray, uChanel: ndarray, vChanel: ndarray) -> ndarray:
    result = np.empty((len(yChanel), len(yChanel[0]),3))
    for i in range(len(yChanel)):
        for j in range(len(yChanel[0])):
            result[i][j][0] = yChanel[i][j]
            result[i][j][1] = uChanel[i][j]
            result[i][j][2] = vChanel[i][j]
    return result

class YUVImage:

    def __init__(self, data: ndarray):
        self.data = data

    # change the YUV channel to RBG channel without using builtin method
    def yuv2bgr(self) -> 'BGRImage':

        result = np.empty((len(self.data), len(self.data[0]), 3))
        for i in range(len(self.data)):

            for j in range(len(self.data[i])):
                y = self.data[i][j][0]
                u = self.data[i][j][1]
                v = self.data[i][j][2]

                r = round(y + 1.13983 * (v - 128))
                g = round(y - 0.39465 * (u - 128) - 0.58060 * (v - 128))
                b = round(y + 2.03211 * (u - 128))

                if r > 255:
                    r = 255
                elif r < 0:
                    r = 0
                if g > 255:
                    g = 255
                elif g < 0:
                    g = 0
                if b > 255:
                    b = 255
                elif b < 0:
                    b = 0

                result[i][j][0] = b
                result[i][j][1] = g
                result[i][j][2] = r

            # print(i, end=' ')

        return BGRImage(result)

    def getYChannel(self) -> ndarray:
        return self.data[:, :, 0]

    def getUChannel(self) -> ndarray:
        return self.data[:, :, 1]

    def getVChannel(self) -> ndarray:
        return self.data[:, :, 2]

    def downSampleY(self) -> ndarray:
        return self.data[::2, ::2, 0]

    def downSampleU(self) -> ndarray:
        return self.data[::4, ::4, 1]

    def downSampleV(self) -> ndarray:
        return self.data[::4, ::4, 2]


class BGRImage:

    def __init__(self, data: ndarray):
        self.data = data

    # change the BGR channel to YUV channel without using builtin method
    def bgr2yuv(self) -> YUVImage:
        result = np.empty((len(self.data), len(self.data[0]), 3))
        for i in range(len(self.data)):

            for j in range(len(self.data[i])):
                b = self.data[i][j][0]
                g = self.data[i][j][1]
                r = self.data[i][j][2]
                # print('b',b,'g',g,'r',r)
                y = round(0.299 * r + 0.587 * g + 0.114 * b)
                u = round(-0.147 * r - 0.289 * g + 0.436 * b + 128)
                v = round(0.615 * r - 0.515 * g - 0.100 * b + 128)
                result[i][j][0] = y
                result[i][j][1] = u
                result[i][j][2] = v
            # print(i, end=' ')

        return YUVImage(result)



if __name__ == '__main__':
    img = cv2.imread('test_image.png')
    cv2.imshow('test', img)
    bgr_img = BGRImage(img)
    #
    # # change the BGR channel to YUV channel
    # cv_img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    my_img_yuv = YUVImage(bgr_img.bgr2yuv().data)
    # my_img_yuv = YUVImage(cv_img_yuv)
    # cv2.imshow('my_img_yuv', my_img_yuv.data)

    print(my_img_yuv.data.shape)




    y_downsample = my_img_yuv.downSampleY()
    u_downsample = my_img_yuv.downSampleU()
    v_downsample = my_img_yuv.downSampleV()




    y_up = bilinearUpsampling(y_downsample, 2,my_img_yuv.data.shape)
    u_up = bilinearUpsampling(u_downsample, 4,my_img_yuv.data.shape)
    v_up = bilinearUpsampling(v_downsample, 4,my_img_yuv.data.shape)


    new_yuv_img = YUVImage(mergeColorChannel(y_up, u_up, v_up))

    cv2.imshow('new_yuv_img', new_yuv_img.data.astype(np.uint8))
    new_bgr_img = new_yuv_img.yuv2bgr()

    cv2.imshow('new_img', new_bgr_img.data.astype(np.uint8))

    print(new_bgr_img.data.shape)
    MSE = 1/(len(img)*len(img[0])*3)*np.sum((img.astype("float") - new_bgr_img.data.astype("float")) ** 2)
    PSNR = 10*math.log10(255/MSE)
    print('MSE', MSE)
    print('PSNR', PSNR)

    cv2.waitKey(0)
