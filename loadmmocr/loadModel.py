from mmocr.apis import MMOCRInferencer

def loadDBNet(device_name):
    ocr=MMOCRInferencer(det='DBnet',device=device_name)
    DBNet=ocr.textdet_inferencer.model
    return DBNet.eval()

def loadCRNN(device_name):
    ocr=MMOCRInferencer(rec='CRNN',device=device_name)
    CRNN=ocr.textrec_inferencer.model
    return CRNN.eval()


def loadmodel_byname(name,device_name):
    if name=="dbnet":
        return loadDBNet(device_name)
    elif name=='crnn':
        return loadCRNN(device_name)
    return None