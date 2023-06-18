import torch
import torch.nn as nn
import numpy as np
from models.backbone import build_resnet
from models.basic import SPP,Conv
from models.loss import compute_loss

class yolov1(nn.Module):
    def __init__(self,device=None,input_size=None,num_classes=20,trainable=False,conf_thresh=0.01,nms_thresh=0.45):
        super().__init__()
        self.device=device
        self.input_size=input_size
        self.trainable=trainable
        self.stride=32
        self.num_classes=num_classes
        self.grid=self.create_grid(input_size)
        self.conf_thresh=conf_thresh
        self.nms_thresh=nms_thresh

        self.backbone,feat=build_resnet(pretrained=self.trainable)
        self.neck=nn.Sequential(
            SPP(),
            Conv(feat*4,feat,k=1)
        )
        self.head=nn.Sequential(
            Conv(feat,feat//2,k=1),
            Conv(feat//2,feat,k=3,p=1),
            Conv(feat,feat//2,k=1),
            Conv(feat//2,feat,k=3,p=1),
        )
        self.pred=nn.Conv2d(feat,1+self.num_classes+4,1)

        if self.trainable:
            self.init_bias()

    def init_bias(self):
            init_prob=.01
            bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
            nn.init.constant_(self.pred.bias[...,:1+self.num_classes],bias_value)

    def create_grid(self,input_size):
         h,w=input_size,input_size
         hs=input_size//self.stride
         ws=input_size//self.stride
         grid_y,grid_x=torch.meshgrid(torch.arange(hs),torch.arange(ws),indexing='ij')  #hs ws 
         grid_xy=torch.stack([grid_x,grid_y],-1)  #hs ws 2 
         grid_xy=grid_xy.view(-1,2).to(self.device).float()
         return grid_xy

    def set_grid(self,input_size):
         self.grid=self.create_grid(input_size)
    
    def inference(self,x):
         with torch.inference_mode():
            pred=self.pred(self.head(self.neck(self.backbone(x))))
            pred=pred.permute(0,2,3,1).flatten(1,2)

            conf_pred=pred[...,0:1] #b hw 1
            class_pred=pred[...,1:self.num_classes+1] #b hw 20 
            txtytwth_pred=pred[...,self.num_classes+1:]

            boxes=self.decode_boxes(txtytwth_pred)/self.input_size #normalize to input size 
            boxes=torch.clamp(boxes,0.,1.)
            
            scores=class_pred.softmax(-1)*conf_pred.sigmoid()

            boxes=boxes.to('cpu').numpy()
            scores=scores.to('cpu').numpy()
            boxes,scores,labels=self.postprocess(boxes,scores) 

               
         return boxes,scores,labels

     #     return conf_pred,class_pred,txtytwth_pred       
     

    def postprocess(self,boxes,scores):
          
          
          scores,labels=np.max(scores,-1),np.argmax(scores,-1)
          batch_size=scores.shape[0]

          
          #first image in batch

          scores=scores[0]
          labels=labels[0]
          boxes=boxes[0]
          keep=scores>0.03

          scores=scores[keep]
          labels=labels[keep]
          boxes=boxes[keep]

          #nms 
          return boxes,scores,labels
             



    def decode_boxes(self,txtytwth_pred): #txty to xyxy
         output=torch.zeros_like(txtytwth_pred)
         txtytwth_pred[...,0:2]=torch.sigmoid(txtytwth_pred[...,0:2])+self.grid
         txtytwth_pred[...,2:4]=torch.exp(txtytwth_pred[...,2:4]) #denormalize

         output[...,0:2]=txtytwth_pred[...,0:2]*self.stride-txtytwth_pred[...,2:4]/2 
         output[...,2:4]=txtytwth_pred[...,0:2]*self.stride+txtytwth_pred[...,2:4]/2
         return output

            
    def forward(self,x):
         if not self.trainable:
              return self.inference(x)
         else:
              pred=self.pred(self.head(self.neck(self.backbone(x))))
              pred=pred.permute(0,2,3,1).flatten(1,2)
              conf_pred=pred[...,:1] #b h*W 1
              class_pred=pred[...,1:self.num_classes+1] #b h*W num_classes 
              txtytwth_pred=pred[...,self.num_classes+1:] #b h*W 4

              return conf_pred,class_pred,txtytwth_pred       
       
if __name__ == "__main__":
    boxes,scores,labels=yolov1(input_size=416,trainable=True).forward(torch.randn(3,3,416,416))
    print(boxes.shape,scores.shape,labels.shape)

     
    