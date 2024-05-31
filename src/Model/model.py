import torch
import torch.nn as nn
from torch.amp import autocast
from torch import einsum
import open_clip
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from typing import List, Optional, Union, Any
from einops import rearrange
import math
from src.utils.utils import exists
from src.utils.utils import default
from src.utils.utils import stable_softmax
from src.utils.utils import expand_mask
from src.utils.utils import decode_question

class BidirectionalCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            heads: int = 8,
            dim_head: int = 64,
            context_dim: Optional[int] = None,
            dropout: float = 0.,
            talking_heads: bool = False,
            prenorm: bool = False,

    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            return_attn: bool = False,
            rel_pos_bias: Optional[torch.Tensor] = None
    
    ):

        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device
        x = self.norm(x)
        context = self.context_norm(context)
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                                           (qk, context_qk, v, context_v))
        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
        attn = stable_softmax(sim, dim=-1)
        context_attn = stable_softmax(sim, dim=-2)
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)
        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim_embedds: int, num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert dim_embedds % num_heads == 0

        self.dim_embedds = dim_embedds
        self.num_heads = num_heads
        self.d_k = dim_embedds // num_heads

        self.W_qkv = nn.Linear(input_dim, 3 * dim_embedds)
        self.W_o = nn.Linear(dim_embedds, dim_embedds)
        self.dropout = nn.Dropout(0)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        self.W_qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_o.weight)
        self.W_o.bias.data.fill_(0)

    def scaled_dot_product_attention(self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            mask: Optional[torch.Tensor] = None
):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            _MASKING_VALUE = -1e+30 if attn_scores.dtype == torch.float32 else -1e+4
            attn_scores = attn_scores.masked_fill(mask == 0, _MASKING_VALUE)
        attention = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attention, V)
        return output, attention

    def combine_heads(self, x: torch.Tensor, batch_size: int, seq_length: int) -> torch.Tensor:
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim_embedds)

    def forward(self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            return_attn: bool = False
):
        batch_size, seq_length, _ = x.size()

        if exists(mask):
            mask = expand_mask(mask)
        QKV = self.W_qkv(x)
        QKV = QKV.reshape(batch_size, seq_length, self.num_heads, 3 * self.d_k)
        QKV = QKV.permute(0, 2, 1, 3)
        q, k, v = QKV.chunk(3, dim=-1)
        attn_output, attention = self.scaled_dot_product_attention(q,k,v, mask)

        output = self.W_o(self.combine_heads(attn_output, batch_size, seq_length))
        if return_attn:
            return output, attention
        return output

class FeedForward(nn.Module):
    def __init__(self, inp_shape: int, output_shape: int, act: nn.Module = nn.ReLU):
        super(FeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(inp_shape, inp_shape*2),
            act(),
            nn.Linear(inp_shape*2, output_shape)
        )
    @autocast("cuda")
    def forward(self, x):
        return self.seq(x)
class TextFeedForward(nn.Module):
    def __init__(self, text_emb_size: int, output_size: int, act: nn.Module = nn.ReLU):
        super(TextFeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(text_emb_size, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, output_size)
        )
    def forward(self, x: torch.Tensor):
        return self.seq(x)

class QFormerBlock(nn.Module):
    def __init__(self, img_emb_size: int, text_emb_size: int, output_size: int, bias: bool = True):
        super(QFormerBlock, self).__init__()
        self.attn = MultiHeadAttention(text_emb_size, text_emb_size, 16)
        self.cross_attn = BidirectionalCrossAttention(
            dim=img_emb_size,
            heads=16,
            dim_head=1024,
            context_dim=text_emb_size
        )
        self.text_feed_forward = TextFeedForward(text_emb_size, output_size)
    @autocast("cuda")
    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        text_emb = self.attn(text_emb)
        img_emb, text_emb = self.cross_attn(img_emb.reshape(-1, 1, img_emb.shape[1]), text_emb)
        text_emb = self.text_feed_forward(text_emb)
        return img_emb, text_emb

class Blocks(nn.Module):
    def __init__(self, img_emb_size: int, text_emb_size: int, n_blocks: int):
        super(Blocks, self).__init__()
        self.model = nn.Sequential(*[QFormerBlock(img_emb_size, text_emb_size, text_emb_size) for _ in range(n_blocks)])
    def forward(self, *x):
        for block in self.model._modules.values():
          x = block(*x)
          if x[0].shape[1] == 1:
            x = (x[0][:, 0, :], x[1])
        return x

class QFormer(nn.Module):
    def __init__(self, img_emb_size: int, text_emb_size: int, output_size: int, n_blocks: int = 4, bias: bool = True):
        super(QFormer, self).__init__()

        self.blocks = Blocks(img_emb_size, text_emb_size, n_blocks)
        self.res = nn.Linear(img_emb_size + text_emb_size, output_size)

    @autocast("cuda")
    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor):
        img_emb, text_emb = self.blocks(img_emb, text_emb)
        text_emb = text_emb.mean(axis=1)
        res_emb = torch.cat((img_emb, text_emb), axis=1)
        res_emb = self.res(res_emb)
        return res_emb

class MLP(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, act: nn.Module = nn.Tanh):
        super(MLP, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_shape, input_shape * 2),
            act(),
            nn.Linear(input_shape * 2, output_shape)
        )

    @autocast("cuda")
    def forward(self, x):
        return self.seq(x)


class BILIP(nn.Module):
    def __init__(self, config: dict, prefix_size: int = 640, dist_loss: Any = nn.MSELoss()):
        super(BILIP, self).__init__()
        self.prefix_length = config.prefix_length
        self.clip_model, _, _ = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm)
        self.llm = T5ForConditionalGeneration.from_pretrained(config.llm)
        
        self.llm_embedding_size = self.llm.get_input_embeddings().weight.shape[1]
        self.clip_project = QFormer(prefix_size, self.llm_embedding_size,
                                    self.llm_embedding_size * self.prefix_length)
        self.device = config.device
        self.dist_loss = dist_loss
        self.mlp = MLP(self.llm_embedding_size, self.llm_embedding_size)

        for p in self.llm.parameters():
            p.requires_grad = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    @autocast("cuda")
    def forward(self, query_tokens: torch.Tensor, query_mask: Optional[torch.Tensor],
                answer_tokens: torch.Tensor, answer_mask: Optional[torch.Tensor], image: Union[torch.Tensor, List[torch.Tensor]]):
        inputs_embeds = self.llm.encoder.embed_tokens(query_tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), inputs_embeds).view(-1, self.prefix_length,
                                                                                   self.llm_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.llm(inputs_embeds=prefix_projections, labels=answer_tokens)
        return out, prefix_projections
    
    def generate(self, image: torch.Tensor, texts: List[str], max_seq_len: int):
        input_tokens = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt").to(self.device)
        inputs_embeds = self.llm.encoder.embed_tokens(input_tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), inputs_embeds).view(-1, self.prefix_length,
                                                                                   self.llm_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.llm.generate(
            inputs_embeds=prefix_projections,
            max_new_tokens=self.prefix_length,
            no_repeat_ngram_size=3,
            repetition_penalty=2.,
        )
        res = [decode_question(x, self.tokenizer) for x in out]
        return res
