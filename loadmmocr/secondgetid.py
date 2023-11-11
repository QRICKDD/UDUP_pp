from mmocr.apis import MMOCRInferencer
from UDUP_pp.tools.imageio import *
import os
abspath="/workspace/mmocr"
ocr=MMOCRInferencer(det='DBNet',rec='CRNN')

inputs=['/workspace/mmocr/demo/demo_text_ocr.jpg']
# img=DBNet_inputload(inputs[0])
# img.requires_grad=True
# img=DBNet_preprocess(img)

ocr.textdet_inferencer(inputs,return_datasamples=True,batch_size=1)
ocr(os.path.join(abspath,'demo/demo_text_ocr.jpg'), show=True, print_result=True)
print("python")




"""
ocr 先进入_call_函数  
    再进入forward
from torchvision import transforms
import cv2
transform = transforms.ToTensor()
image_path='/workspace/mmocr/demo/demo_text_ocr.jpg'
im = cv2.imread(image_path, 1)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
myimg = transform(im)
myimg=transforms.Resize([1024,1220],interpolation=transforms.InterpolationMode.BILINEAR)(myimg)

ocr.textdet_inferencer(inputs,return_datasamples=True,batch_size=1)


 if self.to_float32:
            img = img.astype(np.float32)
        if self.color_type == 'grayscale':
            img = mmcv.image.rgb2gray(img)
        results['img'] = img
        if results.get('img_path', None) is None:
            results['img_path'] = None
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

<class 'mmocr.datasets.transforms.ocr_transforms.Resize'>

mean=torch.tensor([[[123.6750]],[[116.2800]],[[103.5300]]], device='cuda:0')
std=tensor([[[58.3950]],[[57.1200]],[[57.3750]]], device='cuda:0')

整数tensor，
img=(img-mean)/std
"""