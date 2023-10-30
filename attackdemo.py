import os
from mmocr.apis import MMOCRInferencer

from UDUP_pp.tools.imageio import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

torch.backends.cudnn.enabled = False

ctc_loss=nn.CTCLoss()
ce_loss=nn.CrossEntropyLoss()
device_name='cuda:0'

abspath="/workspace/mmocr"
ocr=MMOCRInferencer(rec='CRNN',device=device_name)

img_path='/workspace/mmocr/demo/apple.jpg'
img=CRNN_inputload(img_path)
img=img.to(device_name)

# noise=torch.zeros_like(img)
# noise=noise.to(device_name)
# noise.requires_grad=True

# pre_img=noise+img
#pre_img=pre_img.to(device_name)
pre_img=img
pre_img.requires_grad=True
pre_imgray=CRNN_inputTrans(pre_img,device=device_name)

predict=ocr.textrec_inferencer.model(pre_imgray)
print("ok")

outputs=torch.log_softmax(predict,dim=2)
print("orgin ouput:",torch.max(predict,dim=2))
bsz, seq_len = outputs.size(0), outputs.size(1)
outputs_for_loss = outputs.permute(1, 0, 2).contiguous()

"""
用于测试时候构建target~
"""
tm_v,tm_i=torch.max(outputs,dim=2)
tt=[]
for item in tm_i[0]:
    if item.item()!=36:
        tt.append(item.item())
targets=torch.tensor([tt],dtype=torch.int32).to(device_name)
#targets=torch.tensor([[31, 31]], dtype=torch.int32).to(device_name)
"""
end
"""
targets=torch.tensor([[36]], dtype=torch.int32).to(device_name)
target_lengths = torch.IntTensor([len(t) for t in targets])
target_lengths = torch.clamp(target_lengths, max=seq_len).long().to(device_name)
input_lengths=torch.full(size=(bsz,),fill_value=seq_len,dtype=torch.long)

loss_ctc = ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)
grad = torch.autograd.grad(loss_ctc, pre_img, retain_graph=False, create_graph=False)[0]

#print(grad)

#输入在测试
adv_images = pre_img.detach() - 1 * grad.sign()
adv_images = torch.clamp(adv_images, min=0, max=255).detach()
ocr.textrec_inferencer.model.zero_grad()
adv_images.requires_grad=True



for i in range(200):
    pre_imgray=CRNN_inputTrans(adv_images,device=device_name)
    predict=ocr.textrec_inferencer.model(pre_imgray)
    outputs_for_loss=torch.log_softmax(predict,dim=2)
    print("orgin ouput:",torch.max(predict,dim=2))
    tm_v, tm_i =torch.max(predict, dim=2)
    N=0
    for item in tm_i[0]:
        if item.item() != 36:
            N=N+1
    bsz, seq_len = outputs_for_loss.size(0), outputs_for_loss.size(1)
    outputs_for_loss = outputs_for_loss.permute(1, 0, 2).contiguous()
    # target_CTC
    targets=torch.tensor([[36]*N], dtype=torch.int32).to(device_name)
    target_lengths = torch.IntTensor([len(t) for t in targets])
    target_lengths = torch.clamp(target_lengths, max=seq_len).long().to(device_name)
    input_lengths=torch.full(size=(bsz,),fill_value=seq_len,dtype=torch.long)
    loss_ctc = ctc_loss(outputs_for_loss, targets, input_lengths,target_lengths)
    print(loss_ctc)
    #target_CE
    t_ce=torch.tensor([36],dtype=torch.long)
    t_ce=t_ce.repeat(seq_len).to(device_name)
    loss_ce=ce_loss(predict.squeeze(0),t_ce)
    print(loss_ce)
    #
    #grad = torch.autograd.grad(loss_ctc+t_ce*5, adv_images, retain_graph=False, create_graph=False)[0]
    grad = torch.autograd.grad(loss_ce+10*loss_ctc, adv_images, retain_graph=False, create_graph=False)[0]
    adv_images = adv_images.detach() - 5* grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=255).detach()

    """
    
    """
    if i%9==0:
        print("i=",i)
        numpy_tensor = adv_images.cpu().squeeze(0).permute(1,2,0)
        plt.imshow(numpy_tensor.int())
        plt.show()


    ocr.textrec_inferencer.model.zero_grad()
    adv_images.requires_grad=True


pre_imgray=CRNN_inputTrans(adv_images,device=device_name)
predict=ocr.textrec_inferencer.model(pre_imgray)
print("ok")
outputs=torch.log_softmax(predict,dim=2)
print("orgin ouput:",torch.max(predict,dim=2))
# loss_ctc = ctc_loss(outputs_for_loss, targets, input_lengths,
#                                  target_lengths)
# grad = torch.autograd.grad(loss_ctc, pre_img, retain_graph=False, create_graph=False)[0]
# adv_images = pre_img.detach() + 5 * grad.sign()
# adv_images = torch.clamp(adv_images, min=0, max=255).detach()


