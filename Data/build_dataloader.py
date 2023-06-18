

# import sys
# sys.path.append(r'C:\Users\faroo\yolov1_project\yolov1')


from utils.utils import *
from torch.utils.data import DataLoader

def build_dataloader(args,dataset):
     dataloader=DataLoader(dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                shuffle=True,
                collate_fn=detection_collate_fn)
     
     return dataloader