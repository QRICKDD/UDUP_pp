"""
udup_pp

"""
import os
import torch
import random
import cv2

from mmocr.apis import MMOCRInferencer

from UDUP_pp.Allconfig.Path_Config import eval_vis_path,test_demo_path
from UDUP_pp.Auxiliary.R import *
from UDUP_pp.loadmmocr.loadModel import loadmodel_byname
from UDUP_pp.loaddate.load_main import *
from UDUP_pp.tools.imageio import mmocr_inputTrans, img_tensorshow,img_tensortocv2
from UDUP_pp.tools.Log import logger_config
from UDUP_pp.evaluate.eval_AED import recog_eval
from UDUP_pp.evaluate.Generate_gt import write_into_txt

torch.backends.cudnn.enabled = False

class UDUPpp_Attack():
    def __init__(self, train_root,train_gt_root,test_root,test_gt_root,
                 save_dir,save_gt_res,
                 det_model, rec_model, device_name,
                 miss_or_show='miss',
                 adv_patch_size=(1, 3, 50, 50),
                 eps=40, alpha=3, decay=1.0,
                 T=100, batch_size=20, batch_gt_num=10, gap=20, lambda_rec=1.0):

        self.test_ocr=MMOCRInferencer(det='DBNet',rec='CRNN',device=device_name)


        self.det_model_name = det_model
        if det_model is not None:
            self.det_model = loadmodel_byname(det_model, device_name)
        self.miss_or_show = miss_or_show

        self.rec_model_name = rec_model
        if rec_model is not None:
            self.rec_model = loadmodel_byname(rec_model, device_name)

        self.device_name = device_name
        self.eps, self.alpha, self.decay = eps, alpha, decay
        self.T, self.batch_size, self.batch_gt_num = T, batch_size, batch_gt_num
        self.gap, self.lambda_rec = gap, lambda_rec
        self.adv_patch_size = adv_patch_size

        self.train_root,self.train_gt_root = train_root,train_gt_root
        self.test_root,self.test_gt_root=test_root,test_gt_root
        self.save_dir,self.save_gt_dir=save_dir,save_gt_res
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_gt_dir):
            os.makedirs(self.save_gt_dir)
        self.train_dataset = init_train_dataset(train_root,train_gt_root)
        #self.test_dataset = init_test_dataset(test_root,test_gt_root)

        # all gap
        self.shufflegap = len(self.train_dataset) // self.batch_size
        self.gap = gap

        self.adv_patch = torch.ones(list(adv_patch_size)) * 255
        self.t = 0  # 起始迭代

        self.MseLoss = torch.nn.MSELoss()
        self.CTCLoss = torch.nn.CTCLoss()

        # log 配置
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
            # 迭代取数据集并打乱
            if t % self.shufflegap == 0:
                random.shuffle(self.train_dataset)
                shuff_ti = 0
            batch_dataset = self.train_dataset[shuff_ti * self.batch_size: (shuff_ti + 1) * self.batch_size]
            shuff_ti += 1

            # 一个batch内的梯度和loss
            sum_det_grad = torch.zeros_like(self.adv_patch)  # 采用迭代累加，考虑到内存不够
            sum_rec_grad = torch.zeros_like(self.adv_patch)
            batch_detloss = 0
            batch_recloss = 0

            for [x, gt_path] in batch_dataset:
                """ER
                det_loss
                """
                if self.det_model_name != None:
                    it_adv_patch = self.adv_patch.clone().detach().to(self.device_name)
                    it_adv_patch.requires_grad = True
                    x = x.to(self.device_name)
                    x_r1 = Diverse_module_1(x, t, self.gap)

                    h, w = x_r1.shape[2:]
                    DU = repeat_4D(patch=it_adv_patch, h_real=h, w_real=w)
                    merge_x = DU * m + x_r1 * (1 - m)
                    x_d2 = Diverse_module_2(image=merge_x, now_ti=t, gap=self.gap)
                    # feed into model
                    det_predict = self.det_model.textdet_inferencer.model(x_d2)
                    det_grad, det_loss = self.DetLoss(det_predict, it_adv_patch)
                    # record
                    batch_detloss += det_loss
                    sum_det_grad += det_grad
                    torch.cuda.empty_cache()

                """
                rec_loss
                """
                if self.rec_model_name != None:
                    rec_total_img = x.clone().detach()
                    gt_dataset = load_gt(gt_path)
                    random.shuffle(gt_dataset)
                    batch_len = min(self.batch_gt_num, len(gt_dataset))
                    cal_batch_len=batch_len
                    gt_dataset=gt_dataset[:batch_len]
                    # 抽取十条样本
                    for (box, gt) in gt_dataset:
                        it_adv_patch = self.adv_patch.clone().detach().to(self.device_name)
                        it_adv_patch.requires_grad = True

                        cut_img = get_cut_img(box, rec_total_img).to(self.device_name)
                        #img_tensorshow(cut_img)
                        shake_cut_img = Shake_Box(box, cut_img,self.device_name)
                        m = extract_background(shake_cut_img)
                        h, w = shake_cut_img.shape[2:]
                        DU = repeat_4D_rec(patch=it_adv_patch, h_real=h, w_real=w)
                        merge_img = DU * m + shake_cut_img * (1 - m)

                        target = get_dict_gt(gt,self.rec_model_name).to(self.device_name)
                        if len(target[0])<=2 or w<10:
                            cal_batch_len-=1
                            continue
                        #print("merge_img.shape=",merge_img.shape)
                        trans_img = mmocr_inputTrans(merge_img, self.rec_model_name, self.device_name)
                        #print("trains-img.shape=",trans_img.shape)
                        predict = self.rec_model(trans_img)
                        rec_grad, rec_loss = self.Cal_CTCLoss(predict, target, it_adv_patch)
                        # record
                        batch_recloss += rec_loss
                        sum_rec_grad += rec_grad
                        torch.cuda.empty_cache()

            if self.det_model_name:
                sum_det_grad /= torch.mean(torch.abs(sum_det_grad), dim=(1), keepdim=True)
            if self.rec_model_name:
                sum_rec_grad /= torch.mean(torch.abs(sum_rec_grad), dim=(1), keepdim=True)

            # 一个batch的最终梯度
            grad = sum_det_grad + self.lambda_rec * sum_rec_grad + self.decay * momentum
            momentum = grad

            """
            这里我们会保证所有的loss都是+
            比如detloss一定是target attack，计算时候我直接加上负号，所以更新时候自然变成+
            而recloss目前是无目标攻击，计算梯度后直接+
            所以最终都是+
            """
            temp_patch = self.adv_patch.clone().detach().cpu() + self.alpha * grad
            temp_patch = torch.clamp(temp_patch, min=255 - self.eps, max=255)
            self.adv_patch = temp_patch
            img_tensorshow(self.adv_patch)

            # logger记录
            if cal_batch_len!=0:
                e = "iter:{}, batch_det_loss:{},batch_rec_loss:{},pert:{}==".format(t,
                                                                                    batch_detloss / self.batch_size,
                                                                                    batch_recloss / cal_batch_len,
                                                                                    torch.mean(torch.ones_like(temp_patch)*255
                                                                                           - temp_patch))
            #self.logger.info(e)
            print(e)

            self.demo_test()
            recog_eval(self.test_gt_root,self.save_gt_dir)

            # adv_patch保存
            # temp_save_path = os.path.join(self.save_dir, "advpatch")
            # if not os.path.exists(temp_save_path):
            #     os.makedirs(temp_save_path)
            # save_adv_patch_img(self.adv_patch, os.path.join(temp_save_path, "advpatch_{}.png".format(t)))
            # temp_torch_save_path = os.path.join(self.save_dir, "advtorch")
            # if not os.path.exists(temp_torch_save_path):
            #     os.makedirs(temp_torch_save_path)
            # # 注意这边保存的adv_patch只适配经过mm处理的样本
            # torch.save(self.adv_patch, os.path.join(temp_torch_save_path, "advpatch_{}".format(t)))

    def Cal_DetLoss(self, pred, it_adv_patch):
        if self.miss_or_show == 'miss':
            target = torch.zeros_like(pred)
        else:
            target = torch.ones_like(pred)
        target = target.to(self.device_name)
        loss = self.MseLoss(pred, target)
        grad = torch.autograd.grad(-loss, it_adv_patch, retain_graph=False, create_graph=False, allow_unused=True)[0]
        return grad.detach().cpu(), loss.detach().cpu().item()

    def Cal_CTCLoss(self, pred, target, it_adv_patch):
        pred = torch.log_softmax(pred, dim=2)
        bsz, seq_len = pred.size(0), pred.size(1)
        pred_for_loss = pred.permute(1, 0, 2).contiguous()
        target_lengths = torch.IntTensor([len(t) for t in target])
        target_lengths = torch.clamp(target_lengths, max=seq_len).long().to(self.device_name)
        input_lengths = torch.full(size=(bsz,), fill_value=seq_len, dtype=torch.long)
        ctc_loss = self.CTCLoss(pred_for_loss, target, input_lengths, target_lengths)

        grad = torch.autograd.grad(ctc_loss, it_adv_patch, retain_graph=False, create_graph=False)[0]
        return grad.detach().cpu(), ctc_loss.detach().cpu().item()

    def demo_test(self):
        for item in test_demo_path:
            x=mmocr_imgread(item)
            x = x.to(self.device_name)
            h, w = x.shape[2:]
            m = extract_background(x)
            it_adv_patch = self.adv_patch.clone().detach().to(self.device_name)
            DU = repeat_4D(patch=it_adv_patch, h_real=h, w_real=w)
            merge_x = DU * m + x * (1 - m)
            cv2x=img_tensortocv2(merge_x)
            save_f=os.path.join(self.save_dir,os.path.basename(item))#保存到eval_demo
            cv2.imwrite(save_f,cv2x)
            res=self.test_ocr(save_f, out_dir=eval_vis_path,save_vis=True)#保存到AllData/vis
            torch.cuda.empty_cache()
            #把eval-gt写入self.save_gt_path

            rec_texts = res['predictions'][0]['rec_texts']
            det_polygons = res['predictions'][0]['det_polygons']
            save_gt_path = os.path.join(self.save_gt_dir, os.path.basename(item).split(".")[0] + ".txt")
            write_into_txt(rec_texts, det_polygons, h, w, txt_path=save_gt_path)





