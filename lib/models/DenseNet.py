from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class Densenet169(nn.Module):
    def __init__(self):
        super(Densenet169, self).__init__()
        self.base=torchvision.models.densenet169(pretrained=True)
        #self.base.load_state_dict(torch.load('./densenet169-6f0f7f60.pth'))
        for parma in self.base.parameters():
            parma.requires_grad=False
        self.Pooling=nn.MaxPool2d((10,10))
        self.base.classifier=nn.Sequential(nn.Linear(in_features=1664,out_features=1),
                                           nn.Sigmoid())

    def forward(self,x):
        x1 = self.base.features(x)
        x2=self.Pooling(x1)
        x2=x2.view(x2.size(0),-1)
        y_pred=self.base.classifier(x2)
        return y_pred

class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet,self).__init__()
        self.Conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3)),
                                 nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3),
                                 nn.MaxPool2d(kernel_size=2),
                                 nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
                                 nn.Conv2d(64,128,3),
                                 nn.MaxPool2d((2,2)),
                                 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                                 nn.Conv2d(256, 256, 3),
                                 nn.MaxPool2d((2, 2)),
                                 nn.MaxPool2d((2, 2)))
        self.linear=nn.Sequential(nn.Linear(1024,512),
                                  nn.Linear(512,4),
                                  nn.Sigmoid())

    def forward(self, X):
        x1=self.Conv1(X)
        print("x1.shape")
        print(x1.shape)
        print(x1.size(0))
        x=x1.view(x1.size(0),-1)
        print(x.size())
        x2=x1.view(-1,1024)
        print(x2.shape)
        print(x2.size())
        x3=self.linear(x2)
        print(x3.shape)
        return x3

def main():
    model=Densenet169()
    print(model)

if __name__=='__main__':
    main()


