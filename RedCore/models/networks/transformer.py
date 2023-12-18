
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

#######################################################

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.2, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.ln_12 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("relu", nn.ReLU()),
            ('dropout', nn.Dropout(p=0.1)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        
        self.ln_2 = LayerNorm(d_model)
        self.ln_22 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.dropout = nn.Dropout(p=0.1)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # x = x + self.dropout(self.attention(self.ln_1(x)))
        # x = x + self.dropout(self.mlp(self.ln_2(x)))

        # x = x + self.attention(self.ln_1(x))
        # x = x + self.mlp(self.ln_2(x))

        x = x + self.ln_12(self.attention(self.ln_1(x)))
        x = x + self.ln_22(self.mlp(self.ln_2(x)))
        return x



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, embd_width: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.embd_width = embd_width
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(embd_width, heads, attn_mask) for _ in range(layers)])
        self.proj = nn.Linear(width, embd_width)

        self.attention_vector_weight = nn.Parameter(torch.Tensor(self.embd_width, 1))
        self.attention_layer = nn.Sequential(nn.Linear(self.embd_width, self.embd_width), nn.Tanh())
        self.softmax = nn.Softmax(dim=-1)

        self.muvar = nn.Linear(self.embd_width, self.embd_width * 2)

    def embd_attention(self, x):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文: Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(x)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(x * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, x):
        # embd = self.maxpool(x.transpose(1,2))   # x.size()=>[batch_size, seq_len, hidden_size]
                                                    # x.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = x.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_avgpool(self, x):
        # embd = self.maxpool(x.transpose(1,2))   # x.size()=>[batch_size, seq_len, hidden_size]
                                                    # x.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = x.transpose(1,2)
        embd = F.avg_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)   

    def initialize_parameters(self):
        proj_std = (self.embd_width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.embd_width ** -0.5
        fc_std = (2 * self.embd_width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.resblocks(x)
        x = self.embd_avgpool(x)
        x = torch.sigmoid(x)
        #return z
        x = self.muvar(x).view(-1, 2, self.embd_width)
        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        #z = torch.sigmoid(z)
        return z, mu, log_var




class Transformer2(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, embd_width: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.embd_width = embd_width
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(embd_width, heads, attn_mask) for _ in range(layers)])
        self.proj = nn.Linear(width, embd_width)

        self.attention_vector_weight = nn.Parameter(torch.Tensor(self.embd_width, 1))
        self.attention_layer = nn.Sequential(nn.Linear(self.embd_width, self.embd_width), nn.Tanh())
        self.softmax = nn.Softmax(dim=-1)

        self.muvar = nn.Linear(self.embd_width, self.embd_width * 2)

    def embd_attention(self, x):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文: Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(x)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(x * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, x):
        # embd = self.maxpool(x.transpose(1,2))   # x.size()=>[batch_size, seq_len, hidden_size]
                                                    # x.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = x.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_avgpool(self, x):
        # embd = self.maxpool(x.transpose(1,2))   # x.size()=>[batch_size, seq_len, hidden_size]
                                                    # x.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = x.transpose(1,2)
        embd = F.avg_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)   

    def initialize_parameters(self):
        proj_std = (self.embd_width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.embd_width ** -0.5
        fc_std = (2 * self.embd_width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
    
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.resblocks(x)
        x = self.embd_avgpool(x)
        z = torch.sigmoid(x)
        return z
        # x = self.muvar(x).view(-1, 2, self.embd_width)
        # # get `mu` and `log_var`
        # mu = x[:, 0, :] # the first feature values as mean
        # log_var = x[:, 1, :] # the other feature values as variance
        # # get the latent vector through reparameterization
        # z = self.reparameterize(mu, log_var)
        # #z = torch.sigmoid(z)
        # return z, mu, log_var

