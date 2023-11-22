from mmocr.apis import MMOCRInferencer

def loadDBNet(device_name):
    ocr=MMOCRInferencer(det='DBnet',device=device_name)
    DBNet=ocr.textdet_inferencer.model
    return DBNet.eval()

def loadPSENet(device_name):
    ocr=MMOCRInferencer(det='PSENet',device=device_name)
    PSENet=ocr.textdet_inferencer.model
    return PSENet.eval()

def loadPANet_IC15(device_name):
    ocr=MMOCRInferencer(det='PANet_IC15',device=device_name)
    PANet=ocr.textdet_inferencer.model
    return PANet.eval()

def loadCRNN(device_name):
    ocr=MMOCRInferencer(rec='CRNN',device=device_name)
    CRNN=ocr.textrec_inferencer.model
    return CRNN.eval()



def loadmodel_byname(name,device_name):
    if name=="dbnet":
        return loadDBNet(device_name)
    elif name=='crnn':
        return loadCRNN(device_name)
    elif name=='psenet':
        return loadPSENet(device_name)
    elif name=='panet':
        return loadPANet_IC15(device_name)
    return None