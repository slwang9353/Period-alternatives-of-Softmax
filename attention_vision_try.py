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

def plot_attention_com() -> None:

    row_1 = np.load('Attention_test/norm_sin_softmax/0.239_set/attention_maps.npy')
    row_1 = np.sum(row_1, axis=1).sum(axis=1)
    row_2 = np.load('Attention_test/softmax/0.22_set/attention_maps.npy')
    row_2 = np.sum(row_2, axis=1).sum(axis=1)
    row_3 = np.load('Attention_test/norm_sin_2_max_move/0.256_set/attention_maps.npy')
    row_3 = np.sum(row_3, axis=1).sum(axis=1)
    row_4 = np.load('Attention_test/sin_2_max_move/0.256_set/attention_maps.npy')
    row_4 = np.sum(row_4, axis=1).sum(axis=1)
    rows = [row_2, row_1, row_3, row_4]
    
    plt.figure(dpi=600)
    plt.style.use('ieee')
    img = mpimg.imread('cifar_100_sample.JPEG')
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
            attention_map = cv2.resize(attention_map, (64, 64)) #resize
            normed_map = attention_map / attention_map.max()
            normed_map = (normed_map * 255).astype('uint8')
            normed_map_used = cv2.blur(normed_map,(2,2))  ### blur
            img_used = cv2.resize(img, (64, 64))  ## resize
            img_used = cv2.blur(img_used,(2,2))  ### blur
            plt.imshow(img_used)
            plt.imshow(normed_map_used, alpha=0.4, interpolation='nearest', cmap='seismic')
        plt.subplot(4, 7, (r + 1) * 7)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        img_used = cv2.resize(img, (64, 64))  ## resize
        img_used = cv2.blur(img_used,(2,2))  ### blur
        plt.imshow(img_used)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig('Attention_test.pdf', pad_inches=0)

def plot_attention_com_rose() -> None:
    row_1 = np.load('additional_test/softmax0.375/attention_maps.npy')
    row_1 = np.sum(row_1, axis=1).sum(axis=1)
    print(row_1.shape)
    row_2 = np.load('additional_test/norm_sin_softmax/attention_maps.npy')
    row_2 = np.sum(row_2, axis=1).sum(axis=1).squeeze()
    print(row_2.shape)
    row_3 = np.load('additional_test/sin_2_max_move/attention_maps.npy')
    row_3 = np.sum(row_3, axis=1).sum(axis=1)
    print(row_3.shape)
    row_4 = np.load('additional_test/norm_siren_max/attention_maps.npy')
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





    
