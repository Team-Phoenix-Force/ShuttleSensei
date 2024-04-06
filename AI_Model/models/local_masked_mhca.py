import math
import torch
from torch import nn
from models.layer_norm import LayerNorm
from models.masked_conv1d import MaskedConv1D
from models.weights_init import trunc_normal_

class LocalMaskedMHCA(nn.Module):

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):

        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):

        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        q *= self.scale
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        att = nn.functional.softmax(att, dim=-1)
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask