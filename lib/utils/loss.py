import torch
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self,AT=1000.0,NT=1000.0):
        super(CrossEntropyLoss2d, self).__init__()
        self.AT=torch.FloatTensor([AT])
        self.NT=torch.FloatTensor([NT])
        NT,AT=self.NT,self.AT
        self.wt1=torch.autograd.Variable(NT.abs()/(AT.abs()+NT.abs())).cuda()
        self.wt0=torch.autograd.Variable(AT.abs()/(AT.abs()+NT.abs())).cuda()

    def forward(self,pred,truth):
        Loss=-self.wt1*truth*(pred.log())-self.wt0*(1-truth)*((1-pred).log())
        return Loss.sum()

def main():

    model=torch.nn.Sequential(torch.nn.Linear(10,10),
                              torch.nn.ReLU(),
                              torch.nn.Linear(10,10),
                              torch.nn.ReLU(),
                              torch.nn.Linear(10,1),
                              torch.nn.Sigmoid())

    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    Coss=CrossEntropyLoss2d()

    for epoch in range(10):
        X = torch.autograd.Variable(torch.FloatTensor(torch.randn(10)))
        y = torch.autograd.Variable(torch.FloatTensor(torch.randn(1)))
        y_pred = model(X)
        loss_f = Coss(y_pred, y)
        optimizer.zero_grad()
        loss_f.backward()
        optimizer.step()
        print(y_pred-y)

if __name__ == '__main__':
    main()