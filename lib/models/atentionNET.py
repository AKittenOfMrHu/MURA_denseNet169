import torch

import torch.nn as nn


class ResUnitUp(nn.Module):
    def __init__(self,inch,outch):
        super(ResUnitUp,self).__init__()
        self.unitUp=nn.Sequential(nn.Conv2d(in_channels=inch,out_channels=outch//4,kernel_size=1),
                                nn.Conv2d(in_channels=outch//4,out_channels=outch//4,kernel_size=3,padding=1),
                                nn.Conv2d(in_channels=outch//4,out_channels=outch,kernel_size=1)
                                )

    def forward(self,x):
        x1=self.unitUP(x)

        return x1

class ResUnitDown(nn.Module):
    def __init__(self,inch,outch,enhance=True):
        super(ResUnitDown,self).__init__()
        self.unitDown=nn.Sequential(nn.Conv2d(in_channels=inch,out_channels=outch*4,kernel_size=1),
                                nn.Conv2d(in_channels=outch*4,out_channels=outch*4,kernel_size=3,padding=1),
                                nn.Conv2d(in_channels=outch*4,out_channels=outch,kernel_size=1)
                                )

    def forward(self,x):

        x1=self.unitDown(x)
        return x1


class MaskBranch(nn.Module):
    def __init__(self,ioch):
        super(MaskBranch, self).__init__()
        self.Pooling=nn.MaxPool2d(kernel_size=(2,2))

        self.ResUnit1=ResUnitDown(ioch,ioch//4)
        self.ResUnit2=ResUnitDown(ioch//4,ioch//16)

        self.ResUnit3=ResUnitUp(ioch//16,ioch//4)
        self.ResUnit4=ResUnitUp(ioch//4,ioch)

        self.Upsample=nn.Upsample(scale_factor=2)
        self.Conv=nn.Conv2d(in_channels=ioch,out_channels=ioch,kernel_size=1)

        self.Sigmoid=nn.Sigmoid()


    def forward(self,x):
        x1=self.ResUnit1(self.Pooling(x))
        x2=self.ResUnit2(self.Pooling(x1))
        x3=self.ResUnit3(self.Pooling(x2))
        x4=self.ResUnit4(self.Upsample(x3))
        x5=self.Conv(self.Conv(self.Upsample(x4)))

        x6=self.Sigmoid(x5)
        return x6

class FeatureBranch(nn.Module):#inch=[256,1024],outch=[1024,256]
    def __init__(self,ioch):
        super(FeatureBranch, self).__init__()
        self.ResUnit1=ResUnitUp(ioch,ioch*4)
        self.Poling=nn.MaxPool2d(kernel_size=(2,2))
        self.ResUnit2=ResUnitDown(ioch*4,ioch)

    def forward(self,x):
        x1=self.ResUnit1(x)

        x2=self.ResUnit2(self.Pooling(x1))
        return 0

class AttentionModule(nn.Module):
    def __init__(self,ioch):
        super(AttentionModule, self).__init__()
        self.ResUnit1=ResUnitUp(ioch,ioch*4)

        self.MaskBranch=MaskBranch(ioch*4)
        self.ResUnit2 = ResUnitDown(ioch*4, ioch)
        self.FeatureBranch=FeatureBranch(ioch*4)

    def forward(self,x):
        x1=self.ResUnit1(x)
        x2_1=self.MaskBranch(x1)
        x2_2=self.FeatureBranch(x1)
        #x_t=torch.mul(x2_1,x2_2)
        x2=x2_2+x2_1*x2_2
        x3=self.ResUnit2(x2)

        return x3

class ResidualAtentionNet(nn.Module):
    def __init__(self):
        super(ResidualAtentionNet, self).__init__()
        self.Conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=0)
        self.MaxPooling=nn.MaxPool2d(kernel_size=3,stride=2)
        self.ResUnit1=ResUnitUp(64,256)
        self.Attention1 = AttentionModule(256)
        self.ResUnit2=ResUnitUp(256,512)
        self.Attention2 = AttentionModule(512)
        self.ResUnit3=ResUnitUp(512,1024)
        self.Attention3 = AttentionModule(1024)
        self.ResUnit4 = ResUnitUp(1024, 2048)
        self.AveragePooling=nn.AvgPool2d((7,7))
        self.FC=nn.Sequential(nn.Linear(2048,1024),
                              nn.Relu(),
                              nn.Linear(1024,1),
                              nn.Sigmoid())


        def forward(x):
            C1=self.Conv1(x)
            P1=self.MaxPooling(C1)
            R1=self.ResUnit1(P1)
            A1=self.Attention1(R1)
            # R2=self.ResUnit2(A1)
            #A2=self.Attention2(R2)
            #R3=self.ResUnit3(A2)
            #A3=self.Attention3(R3)
            #R4=self.ResUnit4(A3)
            #P2=self.AveragePooling(R4)
            #y=self.FC(P2)
            return A1


#model= ResUnitUp(16,64)
#model= ResUnitDown(64,4)

#model=MaskBranch(128)
#model=FeatureBranch(128)
model = AttentionModule(1024)
#model=ResidualAtentionNet()

#img=torch.randn(3,229,229)


print(model)
