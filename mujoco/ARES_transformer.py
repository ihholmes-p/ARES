#The ARES transformer is modified mainly from Misha Laskin's code below:
#mainly https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing
#(see also https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA)
#with some help from this by Nikhil Barhate:
#https://github.com/nikhilbarhate99/min-decision-transformer
"""
@misc{laskin2022transformers_rl,
  author       = {Misha Laskin},
  title        = {Transformers for Reinforcement Learning},
  year         = {2022},
  howpublished = {\url{https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv}},
  note         = {Colab notebook linked from tweet: \url{https://twitter.com/MishaLaskin/status/1481767788775628801}}
}
@misc{minimal_decision_transformer,
    author = {Barhate, Nikhil},
    title = {Minimal Implementation of Decision Transformer},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/nikhilbarhate99/min-decision-transformer}},
}
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MaskedCausalAttention(nn.Module):
  def __init__(self, h_dim, max_T, n_heads, drop_p):
    super().__init__()

    self.n_heads = n_heads
    self.max_T = max_T

    self.q_net = nn.Linear(h_dim, h_dim) 
    self.k_net = nn.Linear(h_dim, h_dim) 
    self.v_net = nn.Linear(h_dim, h_dim)

    self.proj_net = nn.Linear(h_dim, h_dim)

    self.att_drop = nn.Dropout(drop_p)
    self.proj_drop = nn.Dropout(drop_p)

    ones = torch.ones((max_T, max_T))
    mask = torch.tril(ones).view(1, 1, max_T, max_T)

    # register buffer makes sure mask does not get updated
    # during backpropagation
    self.register_buffer('mask',mask)

    self.toprint = []

  #test is set to the index of interest for generating the shaped reward
  #i.e. for seeing the reward of timestep 5's state-action pair, set test=4
  def forward(self, x, test=-1):
    B, T, C = x.shape # batch size, seq length, h_dim * n_heads
    N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

    # rearrange q, k, v as (B, N, T, D)
    q = self.q_net(x).view(B, T, N, D).transpose(1,2) 
    k = self.k_net(x).view(B, T, N, D).transpose(1,2)
    v = self.v_net(x).view(B, T, N, D).transpose(1,2)

    # weights (B, N, T, T)
    weights = q @ k.transpose(2,3) / math.sqrt(D)

    # causal mask applied to weights 
    weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
    # normalize weights, all -inf -> 0 after softmax
    normalized_weights = F.softmax(weights, dim=-1)

    if (test > -1):
      l = len(normalized_weights[0][0][-1])
      r = test
      #print("INDEX: ", r)
      for i in range(l):
        if (i != r):
          normalized_weights[0][0][-1][i] = 0
          
    #this is for seeing what the weights look like after the manipulation we perform in the algorithm
    self.toprint = normalized_weights

    # attention (B, N, T, D)

    attention = self.att_drop(normalized_weights @ v)

    # gather heads and project (B, N, T, D) -> (B, T, N*D)
    attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

    out = self.proj_drop(self.proj_net(attention))
    return out


class Block(nn.Module):

  def __init__(self, h_dim, max_T, n_heads, drop_p, internal_embedding):
    super().__init__()

    self.test = -1

    self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)

    self.mlp = nn.Sequential(
            nn.Linear(h_dim, internal_embedding*h_dim),
            nn.GELU(),
            nn.Linear(internal_embedding*h_dim, h_dim),
            nn.Dropout(drop_p),
        )

    self.ln1 = nn.LayerNorm(h_dim)
    self.ln2 = nn.LayerNorm(h_dim)

  def forward(self, x):
    # Attention -> LayerNorm -> MLP -> LayerNorm
    x = x + self.attention(x, self.test) # residual
    x = self.ln1(x) 
    x = x + self.mlp(x) # residual
    x = self.ln2(x)
    return x


class GPT(nn.Module):

  def __init__(self, token_dim, n_blocks, h_dim, max_T, n_heads, drop_p, internal_embedding):
    super().__init__()
    
    # embed input tokens and positions
    # self.proj_state = nn.Embedding(token_dim, h_dim - 12)
    # self.proj_action = nn.Embedding(4, 12)

    self.proj_test = nn.Linear(token_dim, h_dim)
    # parameter = trainable weight matrix 
    init_param_vals = torch.randn(1, max_T, h_dim) / math.sqrt(h_dim)
    self.position_embedding = nn.Parameter(init_param_vals)
    self.dropout = nn.Dropout(drop_p)

    # transformer blocks
    blocks = [Block(h_dim, max_T, n_heads, drop_p, internal_embedding) for _ in range(n_blocks)]
    self.transformer = nn.Sequential(*blocks)

    # projection head
    self.ln = nn.LayerNorm(h_dim)
    self.proj_head = nn.Linear(h_dim, token_dim)

    #LINEAR LAYER FOR MSE
    self.lin = nn.Linear(token_dim, 1)

  def forward(self, x, total_dim, test=-1):
    B, T = x.shape 

    x = x.view(B, T // total_dim, total_dim)
    
    token_h = self.proj_test(x)

    #if you wanted to use a positional encoding, here is where it would go
    #you can see the source cited above for an example of how it's done
    pos_h = 0
    h = token_h + pos_h

    for module in self.transformer:
        module.test = test

    # transformer and prediction
    h = self.ln(self.transformer(h))
    pred = self.proj_head(h)

    #LINEAR LAYER FOR MSE
    pred = self.lin((pred[0])[-1])

    #return pred
    return pred, self.transformer[-1].attention.toprint

  def pred_loss(self, pred, target):
    # pred (B, T, C)  and target (B, T) 
    B, T, C = pred.shape
    return F.cross_entropy(pred.view(B*T, C), target.view(B*T))