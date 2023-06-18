import torch 
from torch import nn
# import sys
# sys.path.append(r'C:\Users\faroo\yolov1_project\yolov1')

# from Data.augment import albumentations
# from Data.dataset import VOCDetection
# from yolo import yolov1
# from Data.grid_creator import gt_creator



class Msewithlogits(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,conf_pred,conf_target):
        conf_pred=torch.sigmoid(conf_pred)
        input=torch.clamp(conf_pred,1e-4,1-1e-4)
        obj_id=(conf_target==1).float()
        noobj_id=(conf_target==0).float()
        obj_loss=obj_id*(input-conf_target)**2
        noob_loss=noobj_id*(input**2)
        return (obj_loss+noob_loss).float()
def compute_loss(conf_pred,class_pred,xywh_pred,targets):
    conf_loss_fn=Msewithlogits()
    class_loss_fn=nn.CrossEntropyLoss(reduction='none')
    xy_loss_fn=nn.BCEWithLogitsLoss(reduction='none')
    wh_loss_fn=nn.MSELoss(reduction='none')

    batch_size=float(targets.size(0))

    conf_pred=conf_pred[...,0] #b hw 
    class_pred=class_pred.permute(0,2,1) #b classes hw 
    xy_pred=xywh_pred[...,:2] #b hw 2
    wh_pred=xywh_pred[...,2:] #b hw 2

    gt_conf=targets[...,0] #b hw 
    gt_class=targets[...,1].long() #b hw
    gt_xy=targets[...,2:4]  #b hw 2
    gt_wh=targets[...,4:6]  #b hw 2
    gt_weight=targets[...,6] #b hw 

    conf_loss=conf_loss_fn(conf_pred,gt_conf)
    conf_loss=conf_loss.sum()/batch_size

    class_loss=class_loss_fn(class_pred,gt_class)*gt_conf
    class_loss=class_loss.sum()/batch_size

    xy_loss=xy_loss_fn(xy_pred,gt_xy).sum(-1)*gt_weight
    xy_loss=xy_loss.sum()/batch_size

    wh_loss=wh_loss_fn(wh_pred,gt_wh).sum(-1)*gt_weight
    wh_loss=wh_loss.sum()/batch_size

    box_loss=xy_loss+wh_loss
    
    total=box_loss+conf_loss+class_loss

    return conf_loss,class_loss,box_loss,total




# if __name__=='__main__':
#     x,y=VOCDetection(r'yolov1/Data/VOCdevkit')[0]
#     y_grid=gt_creator(416,32,[y])
#     conf_pred,class_pred,txtytwth_pred=yolov1(input_size=416,trainable=True).forward(x.unsqueeze(0))
#     print(conf_pred.shape,class_pred.shape,txtytwth_pred.shape)
#     f=compute_loss(conf_pred,class_pred,txtytwth_pred,y_grid)
#     f[-1].backward()
#     print(f)
















