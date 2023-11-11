import os
import math
import cv2
from UDUP_pp.Allconfig.Path_Config import train_data_path,train_gt_path
from mmocr.apis import MMOCRInferencer
#import easyocr

def write_into_txt(rec_texts,det_polygons,h,w,txt_path):
    with open(txt_path,'w') as f:
        for text,ploygon in zip(rec_texts,det_polygons):
            write_content=[]
            for idx,item in enumerate(ploygon):
                if idx%2==0:#偶数是x也就是宽
                    write_content.append(str(max(0,min(math.floor(item),w))))
                else:
                    write_content.append(str(max(0, min(math.floor(item), h))))
            str_write_content=",".join(write_content)
            str_write_content=str_write_content+"\t"+text+"\n"
            f.write(str_write_content)

def load_model_and_predict(device_name,data_path,gt_dir_path):
    file_name_list=os.listdir(data_path)
    file_abs_list=[os.path.join(data_path,item) for item in file_name_list]
    ocr=MMOCRInferencer(det='DBNet',rec="SAR",device=device_name)
    for fp,fn in zip(file_abs_list,file_name_list):
        im=cv2.imread(fp)
        h,w=im.shape[:-1]
        res=ocr(fp)
        rec_texts=res['predictions'][0]['rec_texts']
        det_polygons=res['predictions'][0]['det_polygons']
        gt_name=fn.split(".")[0]+".txt"
        write_into_txt(rec_texts,det_polygons,h,w,txt_path=os.path.join(gt_dir_path,gt_name))

# def gengt():
#     reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
#     result = reader.readtext(os.path.join(train_data_path,"001.png"))
#     print(result)

if __name__=="__main__":
    device_name='cuda:0'
    train_path=train_data_path
    train_gt_path=train_gt_path
    if os.path.exists(train_gt_path)!=True:
        os.makedirs(train_gt_path)

    load_model_and_predict(device_name,train_path,train_gt_path)
    print("aaa")
