import os
from UDUP_pp.tools.imageio import mmocr_imgread


def init_train_dataset(data_root):
    train_dataset=[]
    train_path = os.path.join(data_root, "train")
    train_images = [mmocr_imgread(os.path.join(train_path, name)) for name in os.listdir(train_path)]
    for image in train_images:
        train_dataset.append([image])
    return train_dataset

def init_test_dataset(data_root):
    test_dataset=[]
    test_path = os.path.join(data_root, "test")
    test_gt_path = os.path.join(data_root, "test_craft_gt")
    test_images = [mmocr_imgread(os.path.join(test_path, name)) for name in os.listdir(test_path)]
    test_gts=[os.path.join(test_gt_path,name) for name in os.listdir(test_gt_path)]
    for image,gt in zip(test_images,test_gts):
        test_dataset.append([image,gt])
    return test_dataset