"""
this coding is service for
all about dataset loading

"""
import os
import torch
import numpy as np

import UDUP_pp.Allconfig.Dict_Config as dc
from UDUP_pp.tools.imageio import mmocr_imgread


def init_train_dataset(train_img_path,train_gt_path):
    train_dataset=[]
    train_imgs=os.listdir(train_img_path)
    train_imgs.sort()
    train_img_dataset = [mmocr_imgread(os.path.join(train_img_path, name)) for name in train_imgs]
    train_gts=os.listdir(train_gt_path)
    train_gts.sort()
    train_gt_dataset=[os.path.join(train_gt_path,name) for name in train_gts]
    for image,gt in zip(train_img_dataset,train_gt_dataset):
        train_dataset.append([image,gt])
    return train_dataset

def init_test_dataset(test_img_path,test_gt_path):
    test_dataset=[]
    test_imgs=os.listdir(test_img_path)
    test_imgs.sort()
    test_img_dataset = [mmocr_imgread(os.path.join(test_img_path, name)) for name in test_imgs]
    test_gts=os.listdir(test_gt_path)
    test_gts.sort()
    test_gt_dataset=[os.path.join(test_gt_path,name) for name in test_gts]
    for image,name,gt in zip(test_img_dataset,test_imgs,test_gt_dataset):
        test_dataset.append([image,name,gt])
    return test_dataset

def load_gt(gt_path):
    """
    :return:numpy.darry([[x1,y1,x2,y2],...]) true_label: np.darryy[[12,54,32]]
    """
    with open(gt_path,'r') as f:
        datas=f.readlines()
    res=[]
    for item in datas:
        boxes_str=item.split("\t")[0]
        boxes=np.array([int(pos) for pos in boxes_str.split(",")])
        gt=item.split("\t")[-1].strip()
        res.append((boxes,gt))
    return res

def get_cut_img(box,img)->torch.Tensor:
    x_pos=box[::2]
    y_pos=box[1::2]
    x_max,x_min=max(x_pos),min(x_pos)
    y_max, y_min = max(y_pos), min(y_pos)
    cut_img=img[:,:,y_min:y_max,x_min:x_max]#1,3,h,w
    return cut_img

def get_dict_gt(gt,rec_model_name)->torch.Tensor:
    if rec_model_name=='crnn':
        rec_dict=dc.SRA_dict
    gt_index_list=[]
    keys=rec_dict.keys()
    for item in gt:
        if item not in keys:
            continue
        else:
            gt_index_list.append(rec_dict[item])
    target=torch.tensor([gt_index_list],dtype=torch.int32)
    return target
