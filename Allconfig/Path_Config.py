mmocr_path='/workspace/mmocr'
import os.path as osp
abspath=osp.abspath('..')#/workspace/UDUP_pp


train_data_path=osp.join(abspath,"AllData/train")
train_gt_path=osp.join(abspath,"AllData/train_gt")
test_data_path=osp.join(abspath,"AllData/test")
test_gt_path=osp.join(abspath,"AllData/test_gt")

save_path=osp.join(abspath,"save_path/train_eval")

test_demo_path=[osp.join(abspath,"AllData/test/019.png"),osp.join(abspath,"AllData/test/002.png")]

test_demo_save_path=osp.join(abspath,"AllData/test_demo")