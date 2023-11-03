"""
udup_pp

"""
import os
import torch
import random

from UDUP_pp.Auxiliary.R import *
from UDUP_pp.loadmmocr.loadModel import loadmodel_byname
from UDUP_pp.loaddate.load_main import *
from UDUP_pp.tools.imageio import mmocr_inputTrans,save_adv_patch_img
from UDUP_pp.tools.Log import logger_config

class UDUPpp_Attack():
    def __init__(self,data_root,save_dir,
                 det_model,rec_model,device_name,
                 miss_or_show='miss',
                 adv_patch_size=(1,3,50,50),
                 eps=40,alpha=3,decay=1.0,
                 T=100,batch_size=20,batch_gt_num=10,gap=20,lambda_rec=1.0):
        if det_model!=None:
            self.det_model_name=det_model
            self.det_model=loadmodel_byname(det_model,device_name)
            self.miss_or_show=miss_or_show
        if rec_model!=None:
            self.rec_model_name = rec_model
            self.rec_model=loadmodel_byname(rec_model,device_name)

        self.device_name=device_name
        self.eps,self.alpha,self.decay=eps,alpha,decay
        self.T,self.batch_size,self.batch_gt_num=T,batch_size,batch_gt_num
        self.gap,self.lambda_rec=gap,lambda_rec
        self.adv_patch_size=adv_patch_size

        self.data_root,self.save_dir=data_root,save_dir
        if os.path.exists(self.save_dir) == False:
            os.makedirs(self.save_dir)
        self.train_dataset = init_train_dataset(data_root)
        self.test_dataset = init_test_dataset(data_root)

        # all gap
        self.shufflegap = len(self.train_img_ds) // self.batch_size
        self.gap = gap

        self.adv_patch=torch.ones(list(adv_patch_size))*255
        self.t=0#起始迭代

        self.MseLoss=torch.nn.MSELoss()
        self.CTCLoss=torch.nn.CTCLoss()

        #log 配置
        self.logger = logger_config(log_filename='mylog.log')
        while len(self.logger.handlers) != 0:
            self.logger.removeHandler(self.logger.handlers[0])
        self.logger = logger_config(log_filename='mylog.log')


    def train(self):
        print("start training-====================")

        # 总的用于更新patch的梯度和动量
        momentum = torch.zeros_like(self.adv_patch)

        shuff_ti = 0
        for t in range(self.t + 1, self.T):
            print("iter: ", t)
            #迭代取数据集并打乱
            if t % self.shufflegap == 0:
                random.shuffle(self.train_dataset)
                shuff_ti = 0
            batch_dataset = self.train_dataset[shuff_ti * self.batch_size: (shuff_ti + 1) * self.batch_size]
            shuff_ti += 1

            #一个batch内的梯度和loss
            sum_det_grad = torch.zeros_like(self.adv_patch)#采用迭代累加，考虑到内存不够
            sum_rec_grad = torch.zeros_like(self.adv_patch)
            batch_detloss=0
            batch_recloss=0

            for [x,gt_path] in batch_dataset:
                """
                det_loss
                """
                if self.det_model_name != None:
                    it_adv_patch = self.adv_patch.clone().detach().to(self.device_name)
                    it_adv_patch.requires_grad = True
                    x = x.to(self.device_name)
                    x_r1 = Diverse_module_1(x, t, self.gap)
                    m = extract_background(x_r1)
                    h, w = x_r1.shape[2:]
                    DU = repeat_4D(patch=it_adv_patch, h_real=h, w_real=w)
                    merge_x = DU * m + x_r1 * (1 - m)
                    x_d2 = Diverse_module_2(image=merge_x, now_ti=t, gap=self.gap)
                    # feed into model
                    det_predict = self.det_model.textdet_inferencer.model(x_d2)
                    det_grad,det_loss=self.DetLoss(det_predict,it_adv_patch,self.device_name)
                    # record
                    batch_detloss += det_loss
                    sum_det_grad += det_grad
                    torch.cuda.empty_cache()

                """
                rec_loss
                """
                if self.rec_model_name!=None:
                    gt_dataset= load_gt(gt_path)
                    random.shuffle(gt_dataset)
                    batch_len=min(self.batch_gt_num,len(gt_dataset))
                    # 抽取十条样本
                    for (box, gt) in gt_dataset[:batch_len]:

                        it_adv_patch = self.adv_patch.clone().detach().to(self.device_name)
                        it_adv_patch.requires_grad = True

                        box=Shake_Box(box)
                        cut_img = get_cut_img(box).to(self.device_name)
                        m = extract_background(cut_img)
                        h, w = cut_img.shape[2:]
                        DU = repeat_4D_rec(patch=it_adv_patch, h_real=h, w_real=w)
                        merge_cut_img = DU * m + cut_img * (1 - m)

                        target = get_dict_gt(gt).to(self.device_name)
                        trans_img=mmocr_inputTrans(merge_cut_img,self.rec_model_name,self.device_name)
                        predict=self.rec_model(trans_img)
                        rec_grad,rec_loss=self.CTCLoss(predict,target,it_adv_patch)
                        #record
                        batch_recloss+=rec_loss
                        sum_rec_grad+=rec_grad
                        torch.cuda.empty_cache()

            if self.det_model_name:
                sum_det_grad/=torch.mean(torch.abs(sum_det_grad),dim=(1),keepdim=True)
            if self.rec_model_name:
                sum_rec_grad/=torch.mean(torch.abs(sum_rec_grad),dim=(1),keepdim=True)

            #一个batch的最终梯度
            grad=sum_det_grad+self.lambda_rec*sum_rec_grad+self.decay*momentum
            momentum=grad

            """
            这里我们会保证所有的loss都是+
            比如detloss一定是target attack，计算时候我直接加上负号，所以更新时候自然变成+
            而recloss目前是无目标攻击，计算梯度后直接+
            所以最终都是+
            """
            temp_patch=self.adv_patch.clone().detach().cpu()+self.alpha*grad
            temp_patch = torch.clamp(temp_patch, min=255 - self.eps, max=255)
            self.adv_patch=temp_patch

            #logger记录
            e = "iter:{}, batch_det_loss:{},batch_rec_loss:{},pert:{}==".format(t, batch_detloss / self.batch_size,
                                                                            batch_recloss / self.batch_size,
                                                                            torch.mean(torch.ones_like(temp_patch)
                                                                                       - temp_patch))
            self.logger.info(e)

            #adv_patch保存
            temp_save_path = os.path.join(self.savedir, "advpatch")
            if os.path.exists(temp_save_path) == False:
                os.makedirs(temp_save_path)
            save_adv_patch_img(self.adv_patch, os.path.join(temp_save_path, "advpatch_{}.png".format(t)))
            temp_torch_save_path = os.path.join(self.savedir, "advtorch")
            if os.path.exists(temp_torch_save_path) == False:
                os.makedirs(temp_torch_save_path)
            #注意这边保存的adv_patch只适配经过mm处理的样本
            torch.save(self.adv_patch, os.path.join(temp_torch_save_path, "advpatch_{}".format(t)))


    def DetLoss(self,pred,it_adv_patch):
        if self.miss_or_show == 'miss':
            target = torch.zeros_like(pred)
        else:
            target = torch.ones_like(pred)
        target = target.to(self.device_name)
        loss = self.MseLoss(pred,target)
        grad = torch.autograd.grad(-loss,it_adv_patch, retain_graph=False, create_graph=False, allow_unused=True)[0]
        return grad.detach().cpu(),loss.detach().cpu().item()

    def CTCLoss(self,pred,target,it_adv_patch):
        pred = torch.log_softmax(pred, dim=2)
        bsz, seq_len = pred.size(0), pred.size(1)
        pred_for_loss = pred.permute(1, 0, 2).contiguous()
        target_lengths = torch.IntTensor([len(t) for t in target])
        target_lengths = torch.clamp(target_lengths, max=seq_len).long().to(self.device_name)
        input_lengths = torch.full(size=(bsz,), fill_value=seq_len, dtype=torch.long)
        ctc_loss = self.CTCLoss(pred_for_loss, target, input_lengths, target_lengths)

        grad = torch.autograd.grad(ctc_loss, it_adv_patch, retain_graph=False, create_graph=False)[0]
        return grad.detach().cpu(), ctc_loss.detach().cpu().item()

