import torch
from torchvision import transforms
import random
import numpy as np
def Random_noise(image: torch.Tensor,noise_low,noise_high):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    device=image.device
    temp_image=image.clone().detach().cpu().numpy()
    noise=np.random.uniform(low=noise_low,high=noise_high,size=temp_image.shape)
    noise=torch.from_numpy(noise)
    noise=noise.float()
    noise=noise.to(device)
    image=torch.clamp(image+noise,min=0,max=1)
    return image

def Random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    scale = random.random()
    shape = image.shape
    h, w = shape[-2], shape[-1]
    h, w = int(h * (scale * (high - low) + low)), int(w * (scale * (high - low) + low))
    image = transforms.Resize([h, w])(image)
    return image
def Diverse_module_1(image,now_ti,gap):
    high_index=1.08
    low_index = 0.95
    max_resize_range = (0.80, 1.5)
    pow_num=now_ti//gap
    now_resize_low=max(pow(low_index,pow_num),max_resize_range[0])
    now_resize_high = min(pow(high_index, pow_num),max_resize_range[1])
    resize_image=Random_image_resize(image,low=now_resize_low,high=now_resize_high)
    return resize_image

def Diverse_module_2(image,now_ti,gap):
    #注意这边的扰动~应该是小数点，而不是整数，我读
    resize_image=Diverse_module_1(image,now_ti,gap)
    noise_max=0.05
    noise_start=0.01
    noise_index=1.5
    pow_num=now_ti//gap
    now_noise=min(pow(noise_index,pow_num)*noise_start,noise_max)
    noise_resize_image=Random_noise(resize_image,-1*now_noise,now_noise)
    return noise_resize_image

def extract_background(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_sum = torch.sum(img_tensor, dim=1)
    if torch.max(img_sum)>3:
        mask=(img_sum==255*3)
    else:
        mask = (img_sum == 3)
    mask = mask + 0
    return mask.unsqueeze_(0)

def repeat_4D(patch: torch.Tensor,h_real, w_real) -> torch.Tensor:
    assert (len(patch.shape) == 4 and patch.shape[0] == 1)
    patch_h,patch_w=patch.shape[2:]
    h_num=h_real//patch_h+1
    w_num = w_real // patch_w+1
    patch = patch.repeat(1, 1, h_num, w_num)
    patch = patch[:, :, :h_real, :w_real]
    return patch

def repeat_4D_rec(patch: torch.Tensor,h_real, w_real)-> torch.Tensor:
    patch_h, patch_w = patch.shape[2:]
    h_num = h_real // patch_h + 1
    w_num = w_real // patch_w + 1
    patch = patch.repeat(1, 1, h_num, w_num)
    random_h=random.randint(0,patch.shape[2]-h_real)
    random_w=random.randint(0,patch.shape[-1]-w_real)
    patch=patch[:,:,random_h:random_h+h_real,random_w:random_w+w_real]
    return patch

def Shake_Box(box):
    return box