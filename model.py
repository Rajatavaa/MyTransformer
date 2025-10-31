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
        self.norm = nn.LayerNorm(hidden_dim, eps)

    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.linear_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float):
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
    
    @staticmethod    
    def atten(query,key,value,mask,dropout:nn.Dropout):
        d_k = query.view(-1)
        attention_score = (query @ key.transpose(-2,-1))//(math.sqrt(d_k))@value
        
        if mask is not None:
            attention_score.maked_filled(mask==0,-1e9)
        attention_score = attention_score.softmax(dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value),attention_score
                                                                                  
        
    def forward(self,Q,K,V,mask):
        query = self.w_q(Q)
        key = self.w_k(K)
        value = self.w_v(V)
        
        query = query.view(query.shape(0),query.shape(1),self.h,self.d_k).transpose(1,2) 
        key = key.view(key.shape(0),query.shape(1),self.h,self.d_k).transpose(1,2) 
        value = value.view(value.shape(0),query.shape(1),self.h,self.d_k).transpose(1,2)
        x,self.attention_score = MultiHeadAttention.atten(query,key,value,mask,self.dropout)
        return x.append() 

class ResidualConnection(nn.Module):
    
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)    
        
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    
    def __init__(self,AttentionBlock:MultiHeadAttention,feed_forward_block:FeedForward,d_model:int,dropout:float)->None:
        super().__init__()
        self.AttentionBlock = AttentionBlock
        self.feed_forward_block = feed_forward_block
        self.ResidualConnection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        x = self.ResidualConnection[0](x,lambda x: self.AttentionBlock(x,x,x,src_mask))
        x = self.ResidualConnection[1](x,lambda x: self.feed_forward_block())
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList,d_model:int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.LayerNorm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self,AttentionBlock:MultiHeadAttention,Cross_Attention_Block:MultiHeadAttention,feed_forward_block:FeedForward,d_model:int,dropout:float) -> None:
        super().__init__()    
        self.AttentionBlock = AttentionBlock
        self.Cross_Attention_Block = Cross_Attention_Block
        self.feed_forward_block = feed_forward_block
        self.ResidualConnection = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])
        
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x = self.ResidualConnection[0](x,lambda x: self.AttentionBlock(x,x,x,tgt_mask))
        x = self.ResidualConnection[1](x,lambda x: self.Cross_Attention_Block(x,encoder_output,encoder_output,src_mask))
        x = self.ResidualConnection[1](x,lambda x: self.feed_forward_block())
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList,d_model:int) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.LayerNorm(x)
    
class ProjectionLayer(nn.Module): #This is the Linear layer that is used to map the embedding into the vocabulary
    def __init__(self,d_model,vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)
        
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim = -1)
    
class Transformer(nn.Module):
    def __init__(self, encoder:Encoder,decoder:Decoder,src_embedding:Inputembeddings,tgt_embedding:Inputembeddings,src_position:PositionalEncoding,tgt_position:PositionalEncoding,proj_layer:ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.proj_layer = proj_layer
        
    def encode(self,src,src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(src,src_mask)
    
    def decode(self,tgt,encoder_output,src_mask,tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.proj_layer(x)
    
def transformer_work(self,src_vocab:int,tgt_vocab:int,src_seq_len:int,tgt_seq_len:int,d_model:int=512,N:int=6,h:int=8,dropout:float = 0.1,d_ff:int = 2048)->Transformer:
    src_embed = Inputembeddings(d_model,src_vocab)
    tgt_embed = Inputembeddings(d_model,tgt_vocab)
    src_pos = PositionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model,tgt_seq_len,dropout)
    
    encoder_block = []
    for _ in range(N):
        encoder_block_attention = MultiHeadAttention(d_model,h,dropout)
        feed_forward_block = FeedForward(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_block_attention,feed_forward_block,dropout)
        
    
    
     
