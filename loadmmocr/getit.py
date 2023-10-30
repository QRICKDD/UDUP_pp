from mmocr.apis import MMOCRInferencer
from UDUP_pp.tools.imageio import *
import setGPU
import os
abspath="/workspace/mmocr"
ocr=MMOCRInferencer(rec='NRTR',device='cuda:4')#,rec='CRNN' det='DBNet',

ocr(os.path.join(abspath,'demo/apple.jpg'), show=True, print_result=True)
print("python")




"""
mm-rec 执行流程 NRTR
不同与DBNet这种pipline，它保护四个部分
1.输入文件名，然后mmcv读取，测试过其实等价cv2读取 LoadImageFromFile(ignore_empty=False, min_size=0, to_float32=False, color_type='color', imdecode_backend='cv2', backend_args=None)
2.RescaleToHeight(height=32, min_width=32, max_width=160, width_divisor=16, resize_cfg={'type': 'Resize', 'scale': 0})
3。PadToWidth(width=160, pad_cfg={'type': 'Pad'})
4.PackTextRecogInputs(meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
mean=torch.tensor([[[123.6750]],[[116.2800]],[[103.5300]]], device='cuda:0')
std=tensor([[[58.3950]],[[57.1200]],[[57.3750]]], device='cuda:0')



"""





"""
mm-det预处理流程整理  DBNet
pipline三个流程    实际执行时chrunk函数里会执行pipline流程，走完后进入forward流程
1.输入文件名，然后mmcv读取，测试过其实等价cv2读取
2.计算scale factor 然后cv2.缩放     <class 'mmocr.datasets.transforms.ocr_transforms.Resize'>  缩放用的双线性插值
3.经过mmocr.datasets.transforms.formatting.PackTextDetInputs 
    包括：1把整数数据变成tensor，然后把(h,w,c)通过permute变成(c,h,w)

-----接下来的处理在forward中-----
先进入
mmocr/models/textdet/data_preprocessors/data_preprocessor.py中的forward
在进入basemodel.data_preprocessor.ImgDataPreprocessor中处理
    包括步骤如下:1：norm化，减去均值除以方差 2：stack_batch过程，把size pad 其至32倍数（DBNet）要求 并用0填充

在进入mmocr/models/textdet/detectors/base.py中的BaseTextDector的forward处理，其实会直接送到predict处理
但是输入predict 的不止data，还有data_samples

predict包括路线如下，我们要得到预测的map值，这和应该一样大~
x=extract_feat(inputs)
self.det_head.predict(x,data_samples)




mean=torch.tensor([[[123.6750]],[[116.2800]],[[103.5300]]], device='cuda:0')
std=tensor([[[58.3950]],[[57.1200]],[[57.3750]]], device='cuda:0')

整数tensor，
img=(img-mean)/std
"""



# ocr=MMOCRInferencer(rec='CRNN',device='cuda:4')#,rec='CRNN'
#
# inputs=['/workspace/mmocr/demo/demo_text_ocr.jpg']
# img=DBNet_inputload(inputs[0])
# img.to('cuda:4')
# img.requires_grad=True
# img=DBNet_preprocess(img)
#
# ocr(os.path.join(abspath,'demo/demo_text_ocr.jpg'), show=True, print_result=True)
# print("python")



"""
CRNN
pipline 流程
1：mmocr/datasets/transforms/loading/LoadImageFromFile transform 读取图片
    读取时候采用灰度化读取color_type=grayscale  backend采用cv2  灰度化公式~Gray=R*0.99+G*0.587+B*0.144 
    注意cv2读入的是BGR,不是RGB
2：<class 'mmocr.datasets.transforms.textrecog_transforms.RescaleToHeight'> ->transform函数
    对高度有规定~高度再CRNN中定为32， 宽度则自适应为new_width = math.ceil(float(self.height) / ori_height * ori_width)
3: 经过mmocr.datasets.transforms.formatting.PackTextDetInputs 
    包括：1把整数数据变成tensor，然后把(h,w,c)通过permute变成(c,h,w)
    
接下来的预测流程
第一步执行进入mmocr/models/textrecognizers/base.py中的BaseRecognizer.forward 然后进入predict分支
    包括特征提取（注意extract_feat中的self.with_preprocessor是Fasle 所以不用管）并且self.encoder也是False
    所以CRNN就先走过self.backbone,
        特征提取，输入(1,1,32,32)提取出(1，512，1，9)的特征值
        
        然后走self.decoder.predict 
        decoder 进入mmocr/models/textrecog/decoders/base.BaseDecoder
                经过一层层模型再进入Decoder的{/mmocr/models/textrecog/layers/BidirectionalLSTM},
                    \ 其output输出是[9,1,37] 盲猜37=26+10+''？
                再进入BaseDecoder的postprocessor中->mmocr/models/textrecog/postprocessors/ctc_postprocesor.__call__
                    \ call 中调用父类     /mmocr/models/textrecog/postprocessors/base .__call__
                    \ 然后在父类的call中调用自己的的get_single_prediction其中输入的probs此时是[1,N,37]
                
重新描述下：整个forward函数位于mmocr\models\textrecog\recognizers\encoder_decoder_recognizer.py\EncoderDecoderRecognizer
        [mmocr.models.textrecog.recognizers.crnn.CRNN]
        1.首先执行该类的self.extract_feat(inputs)获得提取特征(1,512,1,9)
        2.然后调用self.decoder.preidct
            self.decoder是[mmocr.models.textrecog.decoders.crnn_decoder.CRNNDecoder']
            predict包括self.forward以及后处理
            整个结构如下：
                self.predict
                    self.forward
                        self.forward_test
                            self.softmax(self.forward_train(feat,None,data_samples=None))
                    self.postprocesseror
        3.最后调用mmocr.models.textrecog.decoders.crnn_decoder.CRNNDecoder.postprocesseror
        这个函数就去掉空符号下标
        36，然后得到概率
self.std=127 ~ std.mean=127 
"""