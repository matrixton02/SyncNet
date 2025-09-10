import torch.nn as nn

class FlexibleNN(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim):
        super(FlexibleNN).__init__()
        layers=[]
        prev_dim=input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            prev_dim=h
        layers.append(nn.Linear(prev_dim,output_dim))
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        x=x.view(x.size(0),-1)
        return self.net(x)