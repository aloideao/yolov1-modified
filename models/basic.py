import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self,c1,c2,k,s=1,p=0,activation=True,**kwargs):
        
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(c1,c2,k,stride=s,padding=p,bias=False,**kwargs),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1,inplace=True) if activation else nn.Identity()
        )
    def forward(self,x):
        return self.conv(x)

class SPP(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        """
        input [b,c,h,w] 
        -> 
        output [b,4c,h,w]
        """
        x_1=F.max_pool2d(x,5,1,2)
        x_2=F.max_pool2d(x,9,1,4)
        x_3=F.max_pool2d(x,13,1,6)
        y=torch.cat([x,x_1,x_2,x_3],dim=1)
        return y 
    

if __name__=="__main__":
    # print(SPP().forward(torch.randn(1,2048,7,7)).shape)
    print(Conv(2048,50,3).forward(torch.randn(1,2048,7,7)).shape)


