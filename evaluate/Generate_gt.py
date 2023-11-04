import os
import math
from UDUP_pp.Allconfig.Path_Config import train_data_path,train_gt_path
from mmocr.apis import MMOCRInferencer
#import easyocr

def write_into_txt(rec_texts,det_polygons,txt_path):
    with open(txt_path,'w') as f:
        for text,ploygon in zip(rec_texts,det_polygons):
            write_content=",".join([str(math.floor(p)) for p in ploygon])
            write_content=write_content+"\t"+text+"\n"
            f.write(write_content)

def load_model_and_predict(device_name,data_path,gt_path):
    ocr=MMOCRInferencer(det='DBNet',rec="SAR",device=device_name)
    res=ocr(os.path.join(train_data_path,"001.png"))
    rec_texts=res['predictions'][0]['rec_texts']
    det_polygons=res['predictions'][0]['det_polygons']
    write_into_txt(rec_texts,det_polygons,txt_path=path)

    print(res)

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

    # gengt()
    print("aaa")
