import torch
import torch.nn as nn
import math 

class Inputembeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_length:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_length, d_model)
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1) #adding a dimension along column
        #arrange is used to create a sequence of numbers
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        pe = pe.unsqueeze(0)
        self.resgister_buffer('pe', pe)
        
        def forward(self,x):
            x = x + (self.pe[:,:x.shape[1],:]).require_grad_(False)
            return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
