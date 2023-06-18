
from Data.dataset import VOCDetection
from evaluator.evaluate import VOCAPIEvaluator

def build_dataset(args,
                  root,
                  device,
                  input_size=None,
                  train_input_size=None,
                  val_input_size=None,
                             ):
    ''''
    root :path to devkit which has the data 
    input_size : default 416 for both train and val 
    mode : transformation mode train or val    
    '''
    if val_input_size is None and train_input_size is None:
        train_input_size=input_size
        val_input_size=input_size


    dataset=VOCDetection(root=root,
                         img_size=train_input_size,
                         mode='val'
                         )
    

    eval=VOCAPIEvaluator(data_root=root,
                         img_size=val_input_size,
                         device=device)
    
    return dataset,eval



