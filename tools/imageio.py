import torch
import cv2
from torchvision import transforms
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

test_img_path='/workspace/mmocr/demo/demo_text_ocr.jpg'


#from UDUP_pp.tools.imageio import img_tensorshow


def img_read(image_path) -> torch.Tensor:
    transform = transforms.ToTensor()
    im = cv2.imread(image_path, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # img_show3(im)
    img = transform(im)
    img = img.unsqueeze_(0)
    return img


def img_tensortocv2(img_tensor: torch.Tensor):
    assert (len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1)
    img_tensor = img_tensor.clone().detach().cpu()
    img_tensor = img_tensor.squeeze()
    #img_tensor = img_tensor.mul_(255).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_tensor = img_tensor.permute(1, 2, 0).type(torch.uint8).numpy()
    img_cv = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)
    return img_cv

def img_tensorshow(img):
    imcv=img_tensortocv2(img)
    plt.imshow(imcv)
    plt.show()

def test_cv_tensor():
    img_path=test_img_path
    img_cv2 = cv2.imread(img_path, 1)
    img_tensor = img_read(img_path)
    img_tensor2cv = img_tensortocv2(torch.unsqueeze(img_tensor, dim=0))
    assert (img_tensor2cv == img_cv2).all()
    print('')

def scale_size(old_w,old_h,task):
    if task=="dbnet":
        scale = (4068,1024)

    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(old_h, old_w),
                       max_short_edge / min(old_h, old_w))
    scale=(scale_factor,scale_factor)
    new_w=int(old_w * float(scale[0]) + 0.5)
    new_h=int(old_h * float(scale[1]) + 0.5)
    return new_w,new_h,scale[0]

def mmocr_imgread(img_path):
    im = cv2.imread(img_path, 1) #读入的图像是(h,w,3)
    im = torch.from_numpy(im)
    im = im.permute(2, 0, 1).contiguous()  #(3,h,w)
    im = im.float()
    return im.unsqueeze(0)

def save_adv_patch_img(adv_patch:torch.Tensor,img_path):
    """
    注意，保存的结果是图片~，但如果直接保存adv_patch 必须结合mmocr——imread流程
    """
    if len(adv_patch.shape)==4:
        adv_patch=adv_patch.sequeeze(0)
    adv_patch=adv_patch.permute(1,2,0).contiguous()
    adv_patch_numpy=adv_patch.numpy()
    adv_patch_numpy.astype(int)
    cv2.imwrite(adv_patch_numpy,img_path)


def DBNet_inputTrans(img,device):
    #assert isinstance(img,torch.Tensor) and img.requires_grad==True and img.device.type=='cuda'
    img = img.squeeze()
    img = img.to(device)

    h, w = img.shape[1:]
    new_w, new_h, scale_factor = scale_size(old_w=w, old_h=h, task='dbnet')
    img = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(img)

    mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    std = torch.Tensor([[[58.3950]], [[57.1200]],[[57.3750]]]).to(device)
    norm_img=(img-mean)/std

    #padding 过程 这是因为很多模型要求输入的shape是32的整数倍
    pad_size=32
    pad_value=0

    tensor_size: torch.Tensor = torch.Tensor([tensor.shape for tensor in [norm_img]])
    max_size=torch.ceil(torch.max(tensor_size, dim=0)[0] / pad_size) * pad_size
    padded_sizes=max_size-tensor_size
    padded_sizes[:, 0] = 0#第一个size是channel 不需要pad
    if padded_sizes.sum()==0:
        return torch.stack(norm_img)
    num_img,dim=1,3
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate([norm_img]):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    result = torch.stack(batch_tensor)
    return result


#从tensor图片读取


#实际情况应该从pytorch的图片读入，所以改写下CRNN_inputload
#然后需要保存梯度~,最后回传应该是传回tensor，因为扰动也是以tensor加上去的
def CRNN_inputTrans(img,device):
    assert isinstance(img, torch.Tensor) and img.device.type == 'cuda'
    imgray = img[0][2] * 0.299 + img[0][1] * 0.587 + img[0][0] * 0.114
    mean,std=torch.tensor([[[127]]]).to(device),torch.tensor([[[127]]]).to(device)
    imgray=(imgray-mean)/std
    """
    启用机 
    """
    h, w = imgray.shape[1:]
    new_h=32
    new_w = math.ceil(float(new_h) / h * w)
    #最小w
    min_w,w_divisor=32,16
    # new_h=min(min_w,new_w)
    if new_w%w_divisor!=0:#对其width
        new_w=round(new_w/w_divisor)*w_divisor
    imgray=imgray.unsqueeze(0)
    imgray = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(imgray)
    return imgray


def NRTR_inputTrans(img,device):

    img = img.squeeze()
    img = img.to(device)

    h, w = img.shape[1:]
    new_h=32
    new_w = math.ceil(float(new_h) / h * w)
    #最小w
    min_w,w_divisor=32,16
    new_w = max(min_w,new_w)
    #最大w
    max_w = 160
    new_w = min(max_w,new_w)

    #new_h=min(min_w,new_w)
    if new_w%w_divisor!=0:#对其width
        new_w=round(new_w/w_divisor)*w_divisor

    img = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(img)

    # padding
    ori_height, ori_width = img.shape[1:]
    pad_value = 0

    tensor_size: torch.Tensor = torch.Tensor([tensor.shape for tensor in [img]])
    max_size = torch.tensor([[3, ori_height, 160]])
    padded_sizes = max_size - tensor_size
    padded_sizes[:, 0] = 0

    num_img, dim = 1, 3
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate([img]):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))

    padded_img = batch_tensor[0]

    mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    std = torch.Tensor([[[58.3950]], [[57.1200]],[[57.3750]]]).to(device)
    norm_img=(padded_img-mean)/std

    norm_img_list = []
    norm_img_list.append(norm_img)

    return torch.stack(norm_img_list)

def PSENet_inputTrans(img,device):

    img = img.squeeze()
    img = img.to(device)

    h, w = img.shape[1:]
    new_w, new_h, scale_factor = scale_size(old_w=w, old_h=h, task='psenet')
    img = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(img)

    mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    std = torch.Tensor([[[58.3950]], [[57.1200]],[[57.3750]]]).to(device)
    norm_img=(img-mean)/std

    #padding 过程 这是因为很多模型要求输入的shape是32的整数倍
    pad_size=32
    pad_value=0

    tensor_size: torch.Tensor = torch.Tensor([tensor.shape for tensor in [norm_img]])
    max_size=torch.ceil(torch.max(tensor_size, dim=0)[0] / pad_size) * pad_size
    padded_sizes=max_size-tensor_size
    padded_sizes[:, 0] = 0#第一个size是channel 不需要pad
    if padded_sizes.sum()==0:
        return torch.stack(norm_img)
    num_img,dim=1,3
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate([norm_img]):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))

    return torch.stack(batch_tensor)

def SAR_inputTrans(img,device):

    img = img.squeeze()
    img = img.to(device)

    h, w = img.shape[1:]
    new_h=48
    new_w = math.ceil(float(new_h) / h * w)
    #最小w
    min_w,w_divisor=48,4
    new_w = max(min_w,new_w)
    #最大w
    max_w = 160
    new_w = min(max_w,new_w)

    #new_h=min(min_w,new_w)
    if new_w%w_divisor!=0:#对其width
        new_w=round(new_w/w_divisor)*w_divisor

    img = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(img)

    #img = img.permute(1,2,0).contiguous()

    #padding
    ori_height, ori_width = img.shape[1:]
    pad_value = 0

    tensor_size: torch.Tensor = torch.Tensor([tensor.shape for tensor in [img]])
    max_size = torch.tensor([[3,ori_height,160]])
    padded_sizes=max_size-tensor_size
    padded_sizes[:, 0] = 0

    num_img, dim = 1, 3
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate([img]):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))

    padded_img = batch_tensor[0]

    mean = torch.Tensor([[[127.0]], [[127.0]], [[127.0]]]).to(device)
    std = torch.Tensor([[[127.0]], [[127.0]],[[127.0]]]).to(device)

    norm_img=(padded_img-mean)/std


    norm_img_list = []
    norm_img_list.append(norm_img)

    return torch.stack(norm_img_list)

def PANet_inputTrans(img,device):

    img = img.squeeze()
    img = img.to(device)

    h, w = img.shape[1:]
    ratio = 1.0
    aspect = 1.0

    short_size = 736
    scale = (ratio * short_size) / min(h, w)

    h_scale = scale * math.sqrt(aspect)
    w_scale = scale / math.sqrt(aspect)
    new_h = round(h * h_scale)
    new_w = round(w * w_scale)

    scale_divisor = 1
    new_h = math.ceil(new_h / scale_divisor) * scale_divisor
    new_w = math.ceil(new_w / scale_divisor) * scale_divisor

    img = transforms.Resize([new_h, new_w], interpolation=transforms.InterpolationMode.BILINEAR)(img)

    #norm
    mean = torch.Tensor([[[123.6750]], [[116.2800]], [[103.5300]]]).to(device)
    std = torch.Tensor([[[58.3950]], [[57.1200]],[[57.3750]]]).to(device)
    norm_img=(img-mean)/std

    #padding
    pad_size=32
    pad_value=0

    tensor_size: torch.Tensor = torch.Tensor([tensor.shape for tensor in [norm_img]])
    max_size=torch.ceil(torch.max(tensor_size, dim=0)[0] / pad_size) * pad_size
    padded_sizes=max_size-tensor_size
    padded_sizes[:, 0] = 0
    if padded_sizes.sum()==0:
        return torch.stack(norm_img)
    num_img,dim=1,3
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate([norm_img]):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))

    return torch.stack(batch_tensor)

def mmocr_inputTrans(img,model_name,device):
    if model_name=='crnn':
        return CRNN_inputTrans(img,device)

    if model_name=='dbnet':
        return DBNet_inputTrans(img,device)

    if model_name=='nrtr':
        return NRTR_inputTrans(img,device)

    if model_name=='psenet':
        return NRTR_inputTrans(img,device)

    if model_name=='sar':
        return SAR_inputTrans(img,device)

    if model_name=='panet':
        return PANet_inputTrans(img,device)




