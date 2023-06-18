from models.yolo import yolov1
def build_yolo(args,device,input_size,trainable=False,num_classes=20):
     model=yolov1(device=device,
                  input_size=input_size,
                  num_classes=num_classes,
                  trainable=trainable)
     
     return model