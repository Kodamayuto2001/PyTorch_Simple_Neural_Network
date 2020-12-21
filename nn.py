import torch.nn.functional as F 
import torch 

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)

    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
