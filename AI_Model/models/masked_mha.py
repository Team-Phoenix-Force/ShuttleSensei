import math
import torch
import torch.nn.functional as F
from torch import nn

# attention / transformers
class MaskedMHA(nn.Module):

    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0   # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):

        B, C, T = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)
        
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask