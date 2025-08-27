import torch
import torch.nn as nn
import math 

class Inputembeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model  #dimension of the model
        self.vocab_size = vocab_size #size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model) #embedding layer
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_length:int,dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length #maximum length of the sequence
        self.dropout = nn.Dropout(dropout) #dropout layer
        
        pe = torch.zeros(seq_length, d_model) #positional encoding matrix
        pos = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)  #adding a dimension along column 
        #arrange is used to create a sequence of numbers
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos*div) #Only the even indices are assigned the sine values
        pe[:,1::2] = torch.cos(pos*div) #Only the odd indices are assigned the cosine values
        pe = pe.unsqueeze(0)
        self.resgister_buffer('pe', pe) #registering the positional encoding matrix as a buffer so that it is not considered a model parameter
        
    def forward(self,x):
        x = x + (self.pe[:,:x.shape[1],:]).require_grad_(False)
        return self.dropout(x)

"""class LayerNorm(nn.Module):
    def __init__(self,eps:float=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias"""
    
class LayerNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        # this creates per-feature gamma (weight) and beta (bias)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)

    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:int):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.linear_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,Q,K,V,mask):
        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)
         