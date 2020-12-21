import numpy as np
import cv2
import glob, os

from data_aug import *

def cvt_gray_reshape_img(img, size=(96, 96)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    img = img.reshape(size[0], size[1], 1)
    return img


img_list = sorted(glob.glob('fingerprint_images/*.jpg'))
# print(len(img_list))

# Data
imgs = np.empty((len(img_list), 96, 96, 1), dtype=np.uint8)
zoom_imgs = np.empty((len(img_list), 96, 96, 1), dtype=np.uint8)
noise_imgs = np.empty((len(img_list), 96, 96, 1), dtype=np.uint8)
h_shifted_imgs = np.empty((len(img_list), 96, 96, 1), dtype=np.uint8)
v_shifted_imgs = np.empty((len(img_list), 96, 96, 1), dtype=np.uint8)
# Label
labels = np.empty((len(img_list), 1), dtype=np.uint16)

for i, img_path in enumerate(img_list):
    img = cv2.imread(img_path)
    imgs[i] = cvt_gray_reshape_img(img)
    # Aug zoom
    zoom_img = zoom(img, 0.7)
    zoom_imgs[i] = cvt_gray_reshape_img(zoom_img)
    # Aug add noise
    noise_img = gaussian_noise(img, var=100)
    noise_imgs[i] = cvt_gray_reshape_img(noise_img)
    # Aug shift horizontally
    h_shifted_img = horizontal_shift(img, ratio=0.2)
    h_shifted_imgs[i] = cvt_gray_reshape_img(h_shifted_img)
    # Aug shift vertically
    v_shifted_img = vertical_shift(img, ratio=0.2)
    v_shifted_imgs[i] = cvt_gray_reshape_img(v_shifted_img)
    # Add labels
    labels[i] = i

# Save data
np.save('dataset/x_real.npy', imgs)
np.save('dataset/x_zoom.npy', zoom_imgs)
np.save('dataset/x_noise.npy', noise_imgs)
np.save('dataset/x_h_shifted.npy', h_shifted_imgs)
np.save('dataset/x_v_shifted.npy', v_shifted_imgs)
# Save label
np.save('dataset/y_real.npy', labels)
