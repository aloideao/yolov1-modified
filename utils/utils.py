import torch 
from thop import profile
import os
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import numpy as np 
import torch
import cv2

#train utils 

#collate function for object detection some target is different for each image 
def detection_collate_fn(batch):
    imgs,labels=list(zip(*batch))
    imgs=[img.float() for img in imgs]
    return torch.stack(imgs,0),labels

def save_folder(path,name):
    path=os.path.join(os.getcwd(),path)
    num=1
    num_of_files=len(os.listdir(path))+1 if os.path.exists(path) else 1 
    for _ in range(num_of_files):
        save_path=os.path.join(path,r'{}{}'.format(name,num))
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
            break
        else:
            num+=1
    print(f'folder saved to {save_path}')
    return save_path
    
    
def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets




def FLOPs_and_Params(model, img_size, device=None):
    x=torch.randn(1,3,img_size,img_size).to(device)
    print('='*50)
    with torch.inference_mode():
        flops, params = profile(model, inputs=(x,))
    print('='*50)
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))



#test utils




VOC_CLASSES = [  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']


class albumentations:
    def __init__(self):
        T=[
                  A.Normalize(
                      mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225],
                      max_pixel_value=255.0,
                  ),
                  ToTensorV2(),
              ]
        
        self.transform=A.Compose(T)
    def __call__(self,img):
        return self.transform(image=img)['image']
    





#plotting copied from repo 
def plot_bbox_labels(img, bboxes, cls_color=(0,255,255), text_scale=0.4):
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        label=VOC_CLASSES[int(bbox[-1])]+f': {bbox[-2]}'
        cls_color=class_colors[int(bbox[-1])]

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=3, thickness=1)[0]
        # plot bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        
        if label is not None:
            # plot title bbox
            cv2.rectangle(img, (x1, y1-t_size[1]+5), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
            # put the test on the title bbox
            cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img  

#letterbox copied from yolov5

def letterbox(im, new_shape=None, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)  

#function to rescale infered bbx to the original image size 
def rescale_bbx(bbx,original_height,original_width,input_size=None):
    bbx[:,0]=bbx[:,0]*original_width
    bbx[:,2]=bbx[:,2]*original_width
    bbx[:,1]=bbx[:,1]*original_height
    bbx[:,3]=bbx[:,3]*original_height
    return bbx 


