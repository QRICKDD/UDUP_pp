"""
this coding is service for
all about dataset loading

"""
import os
import torch
import numpy as np

from UDUP_pp.Allconfig.Dict_Config as dc
from UDUP_pp.tools.imageio import mmocr_imgread


def init_train_dataset(data_root):
    train_dataset=[]
    train_path = os.path.join(data_root, "train")
    train_gt_path = os.path.join(data_root, "train_craft_gt")
    train_images = [mmocr_imgread(os.path.join(train_path, name)) for name in os.listdir(train_path)]
    train_gts = [os.path.join(train_path, name) for name in os.listdir(train_gt_path)]
    for image,gt in zip(train_images,train_gts):
        train_dataset.append([image,gt])
    return train_dataset

def init_test_dataset(data_root):
    #有待修改
    test_dataset=[]
    test_path = os.path.join(data_root, "test")
    test_gt_path = os.path.join(data_root, "test_craft_gt")
    test_images = [mmocr_imgread(os.path.join(test_path, name)) for name in os.listdir(test_path)]
    test_gts=[os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        test_dataset.append([image,gt])
    return test_dataset


def cal_x_y():
    pass

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
    cut_img=img[:,:,x_min:x_max,y_min:y_max]
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
    return rec_dict
