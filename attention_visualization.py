import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np

def attention_trans(attention_map):
    patch = int(np.sqrt(attention_map.shape[0]))
    scores = np.sum(attention_map, axis=0).reshape((patch, patch))
    return scores

def plot_attention_com_rose() -> None:
    row_1 = np.load('Attention_test/softmax/attention_maps.npy')
    row_1 = np.sum(row_1, axis=1).sum(axis=1)
    print(row_1.shape)
    row_2 = np.load('Attention_test/sin_softmax/attention_maps.npy')
    row_2 = np.sum(row_2, axis=1).sum(axis=1).squeeze()
    print(row_2.shape)
    row_3 = np.load('Attention_test/sin_2_max_shifted/attention_maps.npy')
    row_3 = np.sum(row_3, axis=1).sum(axis=1)
    print(row_3.shape)
    row_4 = np.load('Attention_test/norm_siren_max/attention_maps.npy')
    row_4 = np.sum(row_4, axis=1).sum(axis=1).squeeze()
    print(row_4.shape)
    rows = [row_1, row_2, row_3, row_4]
    
    plt.figure()
    plt.style.use('ieee')
    img = mpimg.imread('bw_att_test_sample.JPEG')
    ori_img = mpimg.imread('att_test_sample.JPEG')
    ori_img = img_used = cv2.resize(ori_img, (128, 128))
    for r in range(len(rows)):
        epoch, h, w = rows[r].shape
        for e in range(epoch):
            plt.subplot(len(rows), epoch + 1, 1 + r * (epoch + 1) + e)
            plt.xticks([])
            plt.yticks([])
            if r == 0 and e != 0:
                plt.title(str((e+1) * 5), family= 'Times New Roman')
            if r == 0 and e == 0:
                plt.title('Epochs\n' + str((e+1) * 5), family= 'Times New Roman')
            if r == 0 and e == 0:
                plt.ylabel('Softmax', family= 'Times New Roman')
            if r == 1 and e == 0:
                plt.ylabel('Sin-Softmax', family= 'Times New Roman')
            if r == 2 and e == 0:
                plt.ylabel('Sin2-max_shifted', family= 'Times New Roman')
            if r == 3 and e == 0:
                plt.ylabel('norm-Siren-max', family= 'Times New Roman')            
            attention_map = rows[r][e, :, :]
            attention_map = attention_trans(attention_map)
            attention_map = cv2.resize(attention_map, (128, 128)) #resize
            normed_map = attention_map / attention_map.max()
            normed_map = (normed_map * 255).astype('uint8')
            normed_map_used = cv2.blur(normed_map,(5,5))  ### blur
            img_used = cv2.resize(img, (128, 128))  ## resize
            img_used = cv2.blur(img_used,(5,5))  ### blur
            plt.imshow(img_used, cmap='gray')
            plt.imshow(normed_map_used, alpha=0.65, interpolation='nearest', cmap='seismic')  # 'seismic'
        plt.subplot(len(rows), epoch + 1, 1 + r * (epoch + 1) + e + 1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        img_used = cv2.blur(ori_img,(2,2))  ### blur
        plt.imshow(img_used)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig('ROSE_additional_test.pdf',dpi=600, pad_inches=0)





    
