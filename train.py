import os
import random
import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# import sys
# sys.path.append(r'C:\Users\faroo\yolov1_project\yolov1')

from Data.build_dataset import build_dataset

from models.build import build_yolo
from Data.grid_creator import gt_creator
from utils.utils import *  
from Data.build_dataloader import build_dataloader 
from models.loss import compute_loss

def parse_args():
    parser=argparse.ArgumentParser(description='YOLO OBJECT DETECTION')

    parser.add_argument('--cuda',action='store_true',default=True,
                        help='device is cuda')
    
    parser.add_argument('--save_folder', default=r'yolov1\runs', type=str, 
                        help='Gamma update for SGD')
    
    parser.add_argument('--name', default=r'exp', type=str, 
                        help='name of experiment')
    
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloader')
    
    parser.add_argument('--pin_memory',action='store_true', default=False,
                        help='pin memory used in dataloader')
    
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('-accu', '--accumulate', default=4, type=int, 
                        help='gradient accumulate.')
    
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale FOR LATER') 
    
    
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 200,270], type=int,
                        help='lr epoch to decay')
    
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc for now')
    parser.add_argument('--root', default=r'C:\Users\faroo\yolov1_project\yolov1\Data',
                        help='data root')
    
    parser.add_argument('--resume', default=None,type=str,
                        help='resume training')
    
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')


    parser.add_argument('--epochs', default=300, type=int, 
                        help='number of epochs')

    
    return parser.parse_args()

def train():
    args=parse_args()
    
    print('args :',args)
    print('-'*100)

    save_dir=save_folder(args.save_folder,
                         args.name)
    device='cuda' if args.cuda else 'cpu'

    if args.multi_scale:
        train_size=640
        val_size=416
        input_size=train_size

    else:
        train_size=416
        val_size=416
        input_size=train_size


    if args.dataset =='voc':
        root_path=os.path.join(args.root,'VOCdevkit')


        

    dataset,evaluator=build_dataset(args,root=root_path,
                                    train_input_size=train_size,
                                    val_input_size=val_size,
                                    device=device)
    

    dataloader=build_dataloader(args,dataset)

    ''''
    trainable : yes train mode no inferenece which default 
    grid input size / 32 which will be added to inference txtytwth to xyxyn 

    '''


    model=build_yolo(args,device,input_size=input_size,trainable=True,num_classes=20)
    model.to(device).train()
    model.set_grid(train_size)
    if args.resume is not None:
        print('='*50,'\n\nModel has been loaded'+'\n\n'+'='*50)
        model.load_state_dict(torch.load(args.resume,map_location=device))



    #calculate flops 

    model_copy=deepcopy(model)
    model_copy.trainable=False
    model_copy.eval()
    model_copy.set_grid(val_size)
    FLOPs_and_Params(model_copy,
                     val_size,
                     device=device)
    del model_copy

    
    base_lr=args.lr
    tmp_lr=base_lr
    optimizer= torch.optim.SGD(model.parameters(),
                               lr=base_lr,
                               momentum=args.momentum,
                               weight_decay=args.weight_decay)
    

    batch_nums=len(dataloader)

    epochs=args.epochs 

    def set_lr(optimizer,lr):
        for optim in optimizer.param_groups:
            optim['lr']=lr 


    loop=dataloader
    t0=time.time()
    for epoch in range(epochs):
        if epoch in args.lr_epoch:
            tmp_lr=tmp_lr*0.1
            set_lr(optimizer,tmp_lr)


        for batch_id,(imgs,targets) in enumerate(loop):
            ni=batch_id + epoch*batch_nums
            if not args.no_warm_up:
                if epoch <args.wp_epoch:
                    nw=args.wp_epoch*batch_nums
                    tmp_lr=base_lr*pow(ni/nw,4)
                    set_lr(optimizer,tmp_lr)
                elif epoch==args.wp_epoch and batch_id==0:
                    tmp_lr=base_lr
                    set_lr(optimizer,tmp_lr)
            if args.multi_scale and (batch_id+1)%10==0:
                train_size=random.randint(10,19)*32
                model.set_grid(train_size)
            if args.multi_scale: #batch size 
                imgs=F.interpolate(imgs,train_size,mode='bilinear',align_corners=False)


                
            
            targets=gt_creator(train_size,model.stride,targets)
            imgs=imgs.to(device)
            target=targets.to(device)
            conf_pred,class_pred,txtytwth_pred=model(imgs)
            conf_loss,class_loss,box_loss,total=compute_loss(conf_pred,class_pred,txtytwth_pred,target)
            # total/=args.accumulate
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            
            # if (batch_id+1)%args.accumulate==0:

            if batch_id%10==0:
                print(f'Epoch:{epoch}/{epochs},conf_loss={conf_loss.item()},class_loss={class_loss.item()},box_loss={box_loss.item()},total={total.item()},lr={tmp_lr},ni={ni},train_size={train_size}')


        if (epoch+1)%10==0:
                model.trainable=False
                model.set_grid(val_size)
                model.eval()
                evaluator.evaluate(model)

                model.trainable = True
                model.set_grid(train_size)
                model.train()
                print(evaluator.map)
                torch.save(model.state_dict(),os.path.join(save_dir,f'model_{(epoch+1)/25}_{evaluator.map}.pth'))





if __name__ == '__main__':
    train()
