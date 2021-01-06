import glob, os

from data_aug import *


def cvt_gray_resize_img(img, size=(128, 128)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img

train_path = 'train_data'
img_list = sorted(glob.glob('original_images/*.jpg'))


for i, img_path in enumerate(img_list):
    output_path = train_path
    
    img = cv2.imread(img_path)
    cv2.imwrite(os.path.join(output_path, str(i) + '_original.jpg'), cvt_gray_resize_img(img))
    # Aug zoom
    zoom_img = zoom(img, 0.85)
    zoom_img = cvt_gray_resize_img(zoom_img)
    cv2.imwrite(os.path.join(output_path, str(i) + '_zoom.jpg'), zoom_img)
    # Aug add noise
    noise_img = gaussian_noise(img, var=120)
    noise_img = cvt_gray_resize_img(noise_img)
    cv2.imwrite(os.path.join(output_path, str(i) + '_noise.jpg'), noise_img)
    # Aug shift horizontally
    h_shifted_img = horizontal_shift(img, ratio=0.15)
    h_shifted_img = cvt_gray_resize_img(h_shifted_img)
    cv2.imwrite(os.path.join(output_path, str(i) + '_h_shifted.jpg'), h_shifted_img)
    # Aug shift vertically  
    v_shifted_img = vertical_shift(img, ratio=0.15)
    v_shifted_img = cvt_gray_resize_img(v_shifted_img)
    cv2.imwrite(os.path.join(output_path, str(i) + '_v_shifted.jpg'), v_shifted_img)
    # Aug rotate
    rotated_img = rotate(img, angle=15)
    rotated_img = cvt_gray_resize_img(rotated_img)
    cv2.imwrite(os.path.join(output_path, str(i) + '_rotated.jpg'), rotated_img)