import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.weight_norm import weight_norm

class SimpleClassifier(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim,dropout):
        super(SimpleClassifier,self).__init__()
        layer=[
            weight_norm(nn.Linear(in_dim,hid_dim),dim=None),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim,out_dim),dim=None)
        ]
        self.main=nn.Sequential(*layer)
        
    def forward(self,x):
        logits=self.main(x)
        return logits
    
    
class SingleClassifier(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(SingleClassifier,self).__init__()
        layer=[
            nn.Linear(in_dim,out_dim),
            nn.Dropout(dropout,inplace=True)
        ]
        self.main=nn.Sequential(*layer)
        
    def forward(self,x):
        logits=self.main(x)
        return logits        