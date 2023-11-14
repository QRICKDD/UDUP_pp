mmocr_path='/workspace/mmocr'
import os.path as osp
abspath=osp.abspath('..')#/workspace/UDUP_pp


train_data_path=osp.join(abspath,"AllData/train")
train_gt_path=osp.join(abspath,"AllData/train_gt")
test_data_path=osp.join(abspath,"AllData/test")
test_gt_path=osp.join(abspath,"AllData/test_gt")

save_path=osp.join(abspath,"save_path/train_eval")

test_demo_path=[osp.join(abspath,"AllData/test/019.png"),osp.join(abspath,"AllData/test/002.png")]

eval_demo_save_path=osp.join(abspath,"AllData/eval_demo")
eval_demo_gt_path=osp.join(abspath,"AllData/eval_gt")
eval_vis_path=osp.join(abspath,"AllData")

rec_eval_gt=osp.join(abspath,"evaluate/ADE_eval_data/gt_txts")
rec_eval_rec=osp.join(abspath,"evaluate/ADE_eval_data/recog_txts")