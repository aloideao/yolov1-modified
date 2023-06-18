import numpy as np 
import torch
import cv2
from models.yolo import * 
import random
import argparse
from utils.utils import * 

def parse_args():
    parser=argparse.ArgumentParser(description='YOLO Inference')

    parser.add_argument('--cuda',action='store_true',default=True,
                        help='device is cuda')
    
    parser.add_argument('--weights',default="runs\model_100_0.4420237446847536.pth",
                        type=str,help='model weights')
    

    parser.add_argument('--source',default=0
                        ,help='0 for camera,or path to image')
    
    parser.add_argument('--imgsz',default=416
                        ,help='image size')
    
    parser.add_argument('--conf_threshold',default=.25,
                       type=float,help='conf_threshold')
    
    parser.add_argument('--iou',default=.7,
                        type=float,help='nms threshold')
    
    return parser.parse_args()











args = parse_args()

def test(args=args,transform=albumentations()):
    conf_threshold=args.conf_threshold
    iou_threshold=args.iou
    source_path=args.source
    model_path=args.weights
    size=args.imgsz
    device='cuda' if args.cuda else 'cpu'
    

    model=yolov1(input_size=size,device=device,trainable=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    


    if source_path==0:
        camera = cv2.VideoCapture(0)
        while True:
            return_value, image = camera.read()


            # source=cv2.imread(source_path)[:,:,::-1].astype(np.uint8) 
            source=image.astype(np.uint8)

            height,width,_=source.shape
            img=letterbox(source,new_shape=[size,size])[0] #letterbox to perserve scales 

            input=transform(img).to(device)
            result=model(input.unsqueeze(0))


            #rescaling preds to original size 
            filtered_result=(result[0][result[1]>conf_threshold],
                             result[1][result[1]>conf_threshold],
                             result[2][result[1]>conf_threshold])
            
            filtered_result=np.c_[filtered_result]

            filtered_result[:,:4]=rescale_bbx(filtered_result[:,:4],
                                              height,width)
            
            #apply nms


            ids=torch.ops.torchvision.nms(torch.tensor(filtered_result)[:,:4],
                                          torch.tensor(filtered_result)[:,4],
                                          iou_threshold=iou_threshold)
            
            bbx_rescaled=filtered_result[ids].astype(np.float32)  
            if len(bbx_rescaled.shape)==1:
                bbx_rescaled=bbx_rescaled[None] #unsqueeze to avoid losing dem which can cause an error 

            #plotting
            img=plot_bbox_labels(source,bbx_rescaled)

            cv2.imshow('predict',img)
            if cv2.waitKey(27) & 0xFF == ord('q'):
                break

    elif isinstance(source_path,str):
            
            source=cv2.imread(source_path).astype(np.uint8) 


            height,width,_=source.shape
            img=letterbox(source,new_shape=[size,size])[0] #letterbox to perserve scales 

            input=transform(img).to(device)
            result=model(input.unsqueeze(0))


            #rescaling preds to original size 
            filtered_result=(result[0][result[1]>conf_threshold],
                             result[1][result[1]>conf_threshold],
                             result[2][result[1]>conf_threshold])
            
            filtered_result=np.c_[filtered_result]

            filtered_result[:,:4]=rescale_bbx(filtered_result[:,:4],
                                              height,width)
            
            #apply nms

            ids=torch.ops.torchvision.nms(torch.tensor(filtered_result)[:,:4],
                                          torch.tensor(filtered_result)[:,4],
                                          iou_threshold=iou_threshold)
            
            bbx_rescaled=filtered_result[ids].astype(np.float32)  
            if len(bbx_rescaled.shape)==1:
                bbx_rescaled=bbx_rescaled[None] #unsqueeze to avoid losing dem which can cause an error 

            #plotting
            img=plot_bbox_labels(source,bbx_rescaled)

            #plotting
            img=plot_bbox_labels(source,bbx_rescaled)
            cv2.imshow('predict',img)
            cv2.waitKey(0) 
                

if __name__=='__main__':
    test()
