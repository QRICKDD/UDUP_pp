import os

from UDUP_pp.udup.udup_pp import UDUPpp_Attack
import UDUP_pp.Allconfig.Path_Config as pconfig

ap=os.path.abspath('.')
print(ap)
udup_pp=UDUPpp_Attack(
    train_root=pconfig.train_data_path,train_gt_root=pconfig.train_gt_path,
    test_root=pconfig.test_data_path,test_gt_root=pconfig.test_gt_path,
    save_dir=pconfig.eval_demo_save_path,save_gt_res=pconfig.eval_demo_gt_path,
    det_model=None,rec_model='crnn',device_name='cuda:0',
    miss_or_show='miss',
    adv_patch_size=(1,3,20,20),
    eps=50,alpha=3,decay=0.0,T=100,
    batch_size=20,batch_gt_num=10,gap=20, lambda_rec=1.0)
udup_pp.train()
