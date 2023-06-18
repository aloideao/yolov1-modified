import torch
import numpy as np

def generate_txty(gt_label,height,width,s): #gt_label [cxcywhc] normalized 
    cx,cy,w,h=gt_label[:-1]
    cx_s=cx/s*width #cx n -> cx s 
    cy_s=cy/s*height #cy n -> cy s 

    grid_x=int(cx_s)
    grid_y=int(cy_s)

    tx=cx_s-grid_x
    ty=cy_s-grid_y

    tw=np.log(width*w)
    th=np.log(height*h)
    scale=2-h*w

    return grid_x,grid_y,tx,ty,tw,th,scale


def gt_creator(input_size,stride,label_list=[]):
    h,w=input_size,input_size
    hs=h//stride
    ws=w//stride
    batch_size=len(label_list)
    s=stride
    gt=np.zeros([batch_size,ws,hs,1+4+1+1])
    for batch_id in range(batch_size):
        for gt_label in label_list[batch_id]:
            gt_class=int(gt_label[-1])
            result=generate_txty(gt_label,h,w,s)
            if result:
                grid_x,grid_y,tx,ty,tw,th,scale=result
                gt[batch_id,grid_y,grid_x,0]=1
                gt[batch_id,grid_y,grid_x,1]=gt_class
                gt[batch_id,grid_y,grid_x,2:6]=np.array([tx,ty,tw,th])
                gt[batch_id,grid_y,grid_x,6]=scale
    gt=gt.reshape(batch_size,-1,1+4+1+1)
    return torch.tensor(gt).float() 


if __name__=='__main__':
    z=[[0.68921297,0.65694127, 0.2767208,  0.35649213 ,8.        ],
        [0.60050108 ,0.83609112, 0.29669487, 0.32781776, 8.        ],
        [0.60697715, 0.61676329 ,0.23385132 ,0.29640757, 8.        ]]
    
    print(gt_creator(416,32,[z])[...,2:6].shape)