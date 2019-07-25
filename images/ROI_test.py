"""
author:  xing xiangrui
time:    2018.9.23  10:32

image program to test image ROI

"""

import cv2
import matplotlib.pyplot as plt

# image load
image_name='0_1_f_36.jpg'
ori_img = cv2.imread(image_name)

ROI_idx=[10,352,40,280]
#ROI 
ROI=ori_img[ROI_idx[0]:ROI_idx[1],ROI_idx[2]:ROI_idx[3]]

plt.figure

plt.subplot(311)
plt.imshow(ROI)

ROI_temp=ROI.copy()
ori_img[:,:,:]=0
# show image

plt.subplot(312)
plt.imshow(ori_img)

ori_img[ROI_idx[0]:ROI_idx[1],ROI_idx[2]:ROI_idx[3]]=ROI_temp


#ori_img.save('ROI_image.jpg')
plt.subplot(313)
cv2.imwrite("ROI_img.jpg",ori_img)
plt.imshow(ori_img)


