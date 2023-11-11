from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from mmocr.apis import MMOCRInferencer


mmocr_path='/workspace/mmocr'
import os.path as osp
abspath=osp.abspath('..')#/workspace/UDUP_pp
train_data_path=osp.join(abspath,"AllData/train")


imp=osp.join(train_data_path,"356.png")

img = cv2.imread(imp)
#
# cv2.rectangle(img, (260,56), (340, 126), (0, 255, 0), 3)
# cv2.rectangle(img, (394,464), (401, 469), (0, 255, 0), 3)
#
# # 显示图像
# plt.imshow(img)
# plt.show()


ocr = MMOCRInferencer(det='DBNet', rec="SAR", device='cuda:0')
ocr(imp,show=True)