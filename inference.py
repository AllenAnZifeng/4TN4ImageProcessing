import glob
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import *
from keras.models import *
from tqdm import tqdm

# inference part! for the new image!
model = load_model('model/model2.h5')

for i in range(801, 820):
    img = cv2.imread(f'C:\\Users\\zifen\\Desktop\\4TN4\\Projects\\DIV2K_valid_HR\\0{i}.png')
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # y_channel = img_ycrcb[:, :, 0]

    y = cv2.resize(img_yuv, (64, 64), interpolation=cv2.INTER_AREA)
    y_true = cv2.resize(img_yuv, (128, 128), interpolation=cv2.INTER_AREA)

    y = np.expand_dims(y, axis=0)

    # if you have preprocessing you may want to apply those here!
    y_upsampled = model.predict(y)


    def calculate_psnr(img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))


    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    def calculate_ssim(img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


    PSNR = calculate_psnr(y_true, np.squeeze(y_upsampled[0]))
    SSIM = calculate_ssim(y_true, np.squeeze(y_upsampled[0]))

    fig, axs = plt.subplots(1, 3, figsize=(30, 30))

    axs[0].set_title('down-sampled\n 64*64', fontsize=35)
    axs[0].imshow(cv2.cvtColor(y[0], cv2.COLOR_YUV2BGR))

    axs[1].set_title('up-sampled\n truth 128*128', fontsize=35)
    axs[1].imshow(cv2.cvtColor(y_true, cv2.COLOR_YUV2BGR))
    axs[1].set_xlabel(f'PSNR: Original\n SSIM: 1', fontsize=35)

    axs[2].set_title('predicted\n 128*128', fontsize=35)
    processed_y_upsampled = np.clip(y_upsampled[0], 0, 255).astype(np.uint8)
    predicted_img = cv2.cvtColor(processed_y_upsampled, cv2.COLOR_YUV2BGR)
    # clipped_predicted_img = np.clip(predicted_img, 0, 255).astype(np.uint8)
    axs[2].set_xlabel(f'PSNR: {PSNR:.2f}\n SSIM: {SSIM:.2f}', fontsize=35)
    axs[2].imshow(predicted_img)


    plt.show()
