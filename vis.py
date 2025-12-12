import os
import numpy as np
from scipy.io import loadmat
import cv2
import openslide
# from skimage import measure, io, filters, color
# from skimage.segmentation import clear_border

def softmax(x):
    e_x = np.exp(x - np.max(x))  # 减去最大值以避免数值溢出问题
    return e_x / e_x.sum()

# 初始化路径和参数
src_path = r'/data1/dataset/gbm_data/pipeline_data_usecaptk/fengyy/BrainTumorPath/PCA-EPE/save_/'
namepng = '0006512474F_wsi.png'
number = '6600'
beginNum = 1

namepng = '0006512474G_wsi.png'
number = '7425'
beginNum = 7

namepng = '0007581066A_wsi.png'
number = '4980'
beginNum = 19

namepng = '0006512474J_wsi.png'
number = '8910'
beginNum = 13

name = namepng.split('_')[0]
coord = loadmat(src_path + name + '_' + number + '_coords.mat')['data']
patch_level = loadmat(src_path + name + '_' + number + '_patch_level.mat')['data']
patch_size = int(loadmat(src_path + name + '_' + number + '_patch_size.mat')['data'])

# 读取WSI图像
img = cv2.imread(src_path + namepng)
# _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# mask_inv = cv2.bitwise_not(binary)

for i in range(beginNum, beginNum+6):
    temp = loadmat(src_path + name + '_' + f'{i}' + '_' + number + '_atte.mat')['data']
    temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    map_img = np.zeros([img.shape[0], img.shape[1]])
    for j in range(coord.shape[0]):
        x_start = coord[j, 0]
        y_start = coord[j, 1]
        x_end = x_start + patch_size
        y_end = y_start + patch_size

        # 确保不超出图像边界
        x_end = min(x_end, img.shape[1])
        y_end = min(y_end, img.shape[0])
        map_img[y_start:y_end, x_start:x_end] = 255 * temp[0, j]
        # if np.sum(mask_inv[y_start:y_end, x_start:x_end])/(patch_size*patch_size) < 0:
        #     map_img[y_start:y_end, x_start:x_end] = mean_temp
        # else:
        #     map_img[y_start:y_end, x_start:x_end] = 256 * temp[0, j]

    # map_img = cv2.bitwise_and(map_img, map_img, mask=mask_inv)
    heatmap_i = cv2.applyColorMap(np.uint8(map_img), cv2.COLORMAP_JET)
    cv2.imwrite(src_path + f'{name}_heatmap_{i}.png', heatmap_i)
    ratio = 0.5
    superimposed_img = (heatmap_i * ratio + (1-ratio) * img)
    cv2.imwrite(src_path + f'{name}_map_{i}.png', superimposed_img)

