from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import numpy as np 
import torch
class albumentations:
    def __init__(self,size=416,tranform_mode='train'):
      T_train=[
                  A.Resize(height=size, width=size),
                  A.Rotate(limit=35, p=1.0),
                  A.HorizontalFlip(p=0.5),
                  A.VerticalFlip(p=0.1),
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]

      T_val=[
                  A.Resize(height=size, width=size),
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]
          
      T=T_train if tranform_mode=='train' else  T_val 
      self.transform=A.Compose(T,A.BboxParams('yolo',label_fields=['class_ids']))
    def __call__(self,img,label,class_ids):
      if self.transform: 
         transformed=self.transform(image=img,bboxes=label,class_ids=class_ids)
         return transformed['image'],np.array(transformed['bboxes']),np.array(transformed['class_ids'])
      

if __name__ == "__main__":
   a=albumentations(tranform_mode='train')
   print(a(np.random.randn(224,224,3),label=np.array([[.5,.6,.7,.5]]),class_ids=[[3]]))
