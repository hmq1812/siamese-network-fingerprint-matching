import glob
import cv2

img_list = sorted(glob.glob('original_images/*.jpg'))
num_img = len(img_list)
total_w, total_h = 0, 0

for img_path in img_list:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    total_h += height
    total_w += width

print(total_w/num_img, total_h/num_img)