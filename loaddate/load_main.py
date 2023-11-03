"""
this coding is service for
all about dataset loading

"""
import os
import torch


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
    boxes_list=[]
    gt_list=[]
    res=[(boxes_list[0],gt_list[0])]
    return res

def get_cut_img(box)->torch.Tensor:
    cut_img=torch.Tensor([0])
    return cut_img

def get_dict_gt(gt)->torch.Tensor:
    dict_gt=torch.Tensor([0])
    return dict_gt
