class Config:
    encoder: str = "ViT-B-16"
    # decoder: str = "ai-forever/FRED-T5-large"
    decoder: str = "ai-forever/rugpt3medium_based_on_gpt2"
    batch_size: int = 512
    num_epochs: int = 100
    frozen_gpt: int = 20
    frozen_clip: int = 60
    learning_rate: float  = 2e-4
    save_path: str = "/home/jovyan/vqa_project/baselines/saved_models_FRED-T5-large/"
    prefix_length: int = 50
    only_prefix: int = False
    prefix: str = "prefix_small"
    device: str = "cuda:0"
    save_every: int = 1
    warmup_steps: int = 2000
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.amp import autocast
from torch import einsum
import torch.nn.functional as F

import open_clip

from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers import T5ForConditionalGeneration

from typing import Optional

from transformers.optimization import Adafactor
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchmetrics.text import BLEUScore
from evaluate import load
from statistics import mean
import pandas as pd
# from vqadataset import VQAv2_Dataset
from einops import rearrange
import math
import wandb
import pickle
from accelerate.utils import set_seed

from accelerate import Accelerator

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)
    
def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, dim_embedds, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert dim_embedds % num_heads == 0

        self.dim_embedds = dim_embedds
        self.num_heads = num_heads
        self.d_k = dim_embedds // num_heads

        self.W_qkv = nn.Linear(input_dim, 3 * dim_embedds)
        self.W_o = nn.Linear(dim_embedds, dim_embedds)
        self.dropout = nn.Dropout(0)
        # self.IMAGEFIX = nn.Linear(input_dim, dim_embedds)
        # print("MULTIHEADATTENTION INIT ", input_dim, dim_embedds)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_qkv.weight)
        self.W_qkv.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.W_o.weight)
        self.W_o.bias.data.fill_(0)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            _MASKING_VALUE = -1e+30 if attn_scores.dtype == torch.float32 else -1e+4
            attn_scores = attn_scores.masked_fill(mask == 0, _MASKING_VALUE)
        attention = torch.softmax(attn_scores, dim=-1)
        # print(V)
        # output = torch.matmul(self.dropout(attention), V)
        output = torch.matmul(attention, V)
        return output, attention

    def combine_heads(self, x, batch_size, seq_length):
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim_embedds)

    def forward(self, x, mask=None, return_attn=False):
        batch_size, seq_length, _ = x.size()

        if exists(mask):
            mask = expand_mask(mask)
        # if (x.size() == torch.Size([24, 1, 512])):
          #  print("IMAGEFIX: ")
          #  self.IMAGEFIX(x)
          #  print("DONE")
        QKV = self.W_qkv(x)
        QKV = QKV.reshape(batch_size, seq_length, self.num_heads, 3 * self.d_k)
        QKV = QKV.permute(0, 2, 1, 3)
        q, k, v = QKV.chunk(3, dim=-1)
        attn_output, attention = self.scaled_dot_product_attention(q,k,v, mask)
        # attn_output = attn_output.permute(0, 2, 1, 3)
        # attn_output = attn_output.reshape(batch_size, seq_length, self.dim_embedds)

        output = self.W_o(self.combine_heads(attn_output, batch_size, seq_length))
        if return_attn:
            return output, attention
        return output
class FeedForward(nn.Module): #MLP
    def __init__(self, inp_shape, output_shape, act=nn.ReLU):
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
    def __init__(self, text_emb_size, output_size, act=nn.ReLU):
        super(TextFeedForward, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(text_emb_size, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, text_emb_size*2),
            act(),
            nn.Linear(text_emb_size*2, output_size)
        )
    def forward(self, x):
        return self.seq(x)
class QFormerBlock(nn.Module):
    def __init__(self, img_emb_size, text_emb_size, output_size, bias=True):
        super(QFormerBlock, self).__init__()
        self.attn = MultiHeadAttention(text_emb_size, text_emb_size, 16)
        # self.cross_attn = MultiHeadAttention(img_emb_size, img_emb_size, num_heads=16)
        self.cross_attn = BidirectionalCrossAttention(
            dim=img_emb_size,
            heads=16,
            dim_head=1024,
            context_dim=text_emb_size
        )
        self.text_feed_forward = TextFeedForward(text_emb_size, output_size)

        # print("QFO RMER IMG " , str(img_emb_size), "TEXT", text_emb_size)
    @autocast("cuda")
    def forward(self, img_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        text_emb = self.attn(text_emb)
        # print("QFORMER: да ", img_emb.reshape(-1, 1, img_emb.shape[1]).size(), text_emb.size())
        img_emb, text_emb = self.cross_attn(img_emb.reshape(-1, 1, img_emb.shape[1]), text_emb)
        # print("QFORMER все")
        text_emb = self.text_feed_forward(text_emb)
        return img_emb, text_emb
class Blocks(nn.Module):
    def __init__(self, img_emb_size, text_emb_size, n_blocks):
        super(Blocks, self).__init__()
        self.model = nn.Sequential(*[QFormerBlock(img_emb_size, text_emb_size, text_emb_size) for _ in range(n_blocks)])
    def forward(self, *x):
        for block in self.model._modules.values():
          x = block(*x)
          if x[0].shape[1] == 1:
            x = (x[0][:, 0, :], x[1])
        return x
class QFormer(nn.Module):
    def __init__(self, img_emb_size, text_emb_size, output_size, n_blocks=4, bias=True):
        super(QFormer, self).__init__()

        self.blocks = Blocks(img_emb_size, text_emb_size, n_blocks)
        self.res = nn.Linear(img_emb_size + text_emb_size, output_size)

    @autocast("cuda")
    def forward(self, img_emb, text_emb):
        img_emb, text_emb = self.blocks(img_emb, text_emb)
        text_emb = text_emb.mean(axis=1)
        res_emb = torch.cat((img_emb, text_emb), axis=1)
        res_emb = self.res(res_emb)
        return res_emb

def freeze(
    model,
    freeze_emb=False,
    freeze_ln=False,
    freeze_attn=True,
    freeze_ff=True,
    freeze_other=True,
):
    for name, p in model.named_parameters():
    # freeze all parameters except the layernorm and positional embeddings
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other

    return model
class ClipCaptionModel(nn.Module):
    def __init__(self, config, prefix_length: int, prefix_size: int = 512, dist_loss=nn.MSELoss()):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.clip_model, _, _ = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        self.tokenizer = AutoTokenizer.from_pretrained(config.decoder)
        self.gpt = GPT2LMHeadModel.from_pretrained(config.decoder,
                                                   eos_token_id=self.tokenizer.pad_token_id)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = QFormer(prefix_size, self.gpt_embedding_size,
                                    self.gpt_embedding_size * prefix_length)
        self.device = config.device
        self.dist_loss = dist_loss
        self.mlp = FeedForward(self.gpt_embedding_size, self.gpt_embedding_size)

        for p in self.gpt.parameters():
            p.requires_grad = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    @autocast("cuda")
    def forward(self, query_tokens: torch.Tensor, query_mask: Optional[torch.Tensor],
                answer_tokens: torch.Tensor, answer_mask: Optional[torch.Tensor], image):
        embedding_text = self.gpt.transformer.wte(query_tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), embedding_text).view(-1, self.prefix_length,
                                                                                   self.gpt_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.gpt(inputs_embeds=prefix_projections, labels=answer_tokens)
        return out, prefix_projections

    def generate(self, image, texts, max_seq_len):
        # tokens = torch.tensor(self.tokenizer.batch_encode_plus(texts, )['input_ids'], dtype=torch.int64).to(self.device)
        tokens = torch.tensor(self.tokenizer.batch_encode_plus(texts, padding='max_length', max_length=max_seq_len, truncation=True)['input_ids'], dtype=torch.int64).to(self.device)
        embedding_text = self.gpt.transformer.wte(tokens)
        image = self.clip_model.encode_image(image)
        prefix_projections = self.clip_project(image.float(), embedding_text).view(-1, self.prefix_length,
                                                                                   self.gpt_embedding_size)
        prefix_projections = self.mlp(prefix_projections)
        out = self.gpt.generate(
            inputs_embeds=prefix_projections,
            max_new_tokens=self.prefix_length,
            no_repeat_ngram_size=3,
            repetition_penalty=2.,
        )
        res = [decode_question(x, self.tokenizer) for x in out]
        return res

bertscore = load("bertscore")
meteor = load('meteor')
rouge = load('rouge')
bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3]] + [bertscore, meteor, rouge]

wandb.login(key="278590c2621521efe866317352d7f3e13fef885f")
wandb.init(project="", sync_tensorboard=True, name="")

def decode_question(question_token, tokenizer):
    decoded_string = tokenizer.decode(question_token)
    # if "<pad>" in decoded_string:
    #     truncate_pads = decoded_string.index("<pad>")
    #     decoded_string = decoded_string[:truncate_pads]
    decoded_string = decode_question.replace("<pad>", "")
    return decoded_string

def training_loop(mixed_precision="fp16", seed:int=42, args=None, train_dataset=None, val_dataset=None):
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size
    }
    set_seed(seed)
    # Initialize accelerator
    # accelerator = Accelerator(mixed_precision=mixed_precision)
    accelerator = Accelerator()
    args.device = accelerator.device
    # Build dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=20, shuffle=True, drop_last=False)
    eval_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=20, shuffle=True, drop_last=False)
    # train_dataloader, eval_dataloader = get_dataloaders(batch_size)
    
    # instantiate the model (we build the model here so that the seed also controls new weight initaliziations)
    model = ClipCaptionModel(args, args.prefix_length)

    # We normalize the batches of images to be a bit faster
    # mean = torch.tensor(model.default_cfg["mean"])[None, :, None, None]
    # std = torch.tensor(model.default_cfg["std"])[None, :, None, None]
    
    # To make this constant available on the active device, we set it to the accelerator device
    # mean = mean.to(accelerator.device)
    # std = std.to(accelerator.device)
    
    # Intantiate the optimizer
    optimizer = Adafactor(params=model.parameters(), lr=config.learning_rate,
                          relative_step=False
                          )
    # Instantiate the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15000
    )
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    
    # Now we train the model
    print("Start train model")
    for epoch in range(args.num_epochs):

        if epoch == args.frozen_gpt:
            print("LLM UNFROZEN")
            for p in model.gpt.parameters():
                p.requires_grad = True
        if epoch == args.frozen_clip:
            print("CLIP UNFROZEN")
            for p in model.clip_model.parameters():
                p.requires_grad = True
        
        # train(model, optimizer, scheduler, loss_func, train_loader, epoch, args)
        # evaluate(model, optimizer, scheduler, loss_func, val_loader, args)

        print(f"---------- Train epoch {epoch} ---------")
        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in enumerate(pbar):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            query_tokens, query_mask, prefix = query_tokens.to(accelerator.device), query_mask.to(accelerator.device), prefix.to(
            accelerator.device, dtype=torch.bfloat16)
            answer_tokens, answer_mask = answer_tokens.to(accelerator.device), answer_mask.to(accelerator.device)

            outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
            logits = outputs.logits
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                    ignore_index=0)
            loss2 = model.dist_loss(model.gpt.transformer.wte(answer_tokens).to(torch.float32), proj.to(torch.float32))
            loss += loss2
            # loss.backward()

            # inputs = (batch["image"] - mean) / std
            # outputs = model(inputs)
            # loss = torch.nn.functional.cross_entropy(outputs, batch["label"])
                    #backpropogation
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item(), "dist_loss": loss2.item()})
            wandb.log({"loss": loss.item(), "dist_loss": loss2.item()})

            if step % 1000 == 0:
                print("TEXT:", decode_question(answer_tokens[0]))
                print("PREDICTED: ", model.generate(torch.tensor([train_dataset[idx[0]][4].tolist()]).to(args.device),
                                                    [decode_question(query_tokens[0], model.tokenizer)], train_dataset.max_seq_len)[0])
        with open(f'{args.save_path}checkpoint_{epoch}.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"---------- Evaluate epoch {epoch} ---------")
        model.eval()
        pbar = tqdm(eval_dataloader, total=len(eval_dataloader))
        accurate = 0
        num_elems = 0
        
        bl1 = []
        bl2 = []
        bl3 = []
        brt = []
        mtr = []
        rg = []
        val_losses = []
        val_dist = []
        for step, (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in enumerate(pbar):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            query_tokens, query_mask, prefix = query_tokens.to(accelerator.device), query_mask.to(accelerator.device), prefix.to(
            accelerator.device, dtype=torch.bfloat16)
            answer_tokens, answer_mask = answer_tokens.to(accelerator.device), answer_mask.to(accelerator.device)
            
            # inputs = (batch["image"] - mean) / std
            
            with torch.no_grad():
                outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
            logits = outputs.logits
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                    ignore_index=0)
            loss2 = model.dist_loss(model.gpt.transformer.wte(answer_tokens), proj)
            
            real = [decode_question(answer_tokens[i], model.tokenizer) for i in range(len(answer_tokens))]
            pred = model.generate(torch.tensor([val_dataset[idx[j]][4].tolist() for j in range(len(idx))]).to(accelrator.device),
                              [decode_question(query_tokens[j], model.tokenizer) for j in range(len(idx))], val_dataset.max_seq_len)
    
            # predictions = outputs.argmax(dim=-1)
            # accurate_preds = accelerator.gather(predictions) == accelerator.gather(batch["label"])
            # num_elems += accurate_preds.shape[0]
            # accurate += accurate_preds.long().sum()
            bl1.append(bleu_scorers[0](pred, real))
            bl2.append(bleu_scorers[1](pred, real))
            bl3.append(bleu_scorers[2](pred, real))
            brt.append(bleu_scorers[3].compute(predictions=pred, references=real, lang="ru")['f1'])
            mtr.append(bleu_scorers[4].compute(predictions=pred, references=real)['meteor'])
            rg.append(bleu_scorers[5].compute(predictions=pred, references=real)['rougeL'])

            if step % 400 == 0:
                print("TEXT:", real[0])
                print("PREDICTED: ", pred[0])

                imgs = []
                for j in range(len(idx)):
                    wa_img = wandb.Image(
                        val_dataset.get_image(idx[j]),
                        caption=f"REAL : {real[j]}, PREDICTED : {pred[j]}"
                    )
                    imgs.append(wa_img)

                wandb.log({"Generations.": imgs})
            pbar.set_postfix({"val_loss": loss.item(), "val_dist": loss2.item()})
            val_losses.append(loss.item())
            val_dist.append(loss2.item())

        wandb.log({"val_loss": mean(val_losses),
               "val_dist": mean(val_dist)})
        # wandb.log({
        #     "bleu_1": mean([tensor.item() for tensor in bl1]),
        #     "bleu_2": mean([tensor.item() for tensor in bl2]),
        #     "bleu_3": mean([tensor.item() for tensor in bl3]),
        #     "bert_score": np.mean(np.mean([tensor for tensor in brt])),
        #     "meteor_score": np.mean([tensor for tensor in mtr]),
        #     "rouge_score": np.mean([tensor for tensor in rg])
        # })

        # eval_metric = accurate.item() / num_elems
        # # Use accelerator.print to print only on the main process.
        # accelerator.print(f"epoch {epoch}: {100 * eval_metric:.2f}")

class BidirectionalCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            heads=8,
            dim_head=64,
            context_dim=None,
            dropout=0.,
            talking_heads=False,
            prenorm=False,
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
            x,
            context,
            mask=None,
            context_mask=None,
            return_attn=False,
            rel_pos_bias=None
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

from torch.utils.data import Dataset
import sys
from matplotlib import pyplot as plt
import json
from PIL import Image
class VQAv2_Dataset(Dataset):
    def __init__(self, config, dataset_path, coef_size=0.1,
                 tokenizer_name="", prefix_length=20, normalize_prefix=False, imagespath_split=None):
        if not tokenizer_name:
            tokenizer_name = config.decoder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        clip_model, _, self.preprocess = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(dataset_path, 'r') as f:
            dataset = json.loads(list(f)[0])

        self.img_paths = []
        self.query_tokens = []
        self.answer_tokens = []

        max_img = len(dataset)*coef_size
        for i, el in tqdm(enumerate(dataset), total=max_img):
            answer = el['answer'] 
            question = el['question']
            self.query_tokens += [torch.tensor(self.tokenizer.encode(question), dtype=torch.int64)]
            self.answer_tokens += [torch.tensor(self.tokenizer.encode(answer), dtype=torch.int64)]
            if ("val" in imagespath_split):
                self.img_paths += [imagespath_split + el['image_id'].replace("train", "val") + ".jpg"]
            else:
                self.img_paths += [imagespath_split + el['image_id'] + ".jpg"]
            if int(i) >= max_img:
                  break
        del dataset
        sys.stdout.flush()

        #all_len
        self.max_seq_len = prefix_length
        # self.type = data_type

    """Почему не паддили captions?"""
    def pad_tokens(self, item: int):
        query_tokens = self.query_tokens[item]
        padding = self.max_seq_len - query_tokens.shape[0]
        if padding > 0:
            query_tokens = torch.cat((query_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.query_tokens[item] = query_tokens
        elif padding < 0:
            query_tokens = query_tokens[:self.max_seq_len]
            self.query_tokens[item] = query_tokens
        query_mask = query_tokens.ge(0)  # mask is zero where we out of sequence
        query_tokens[~query_mask] = 0
        query_mask = query_mask.float()


        answer_tokens = self.answer_tokens[item]
        padding = self.max_seq_len - answer_tokens.shape[0]
        if padding > 0:
            answer_tokens = torch.cat((answer_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.answer_tokens[item] = answer_tokens
        elif padding < 0:
            answer_tokens = answer_tokens[:self.max_seq_len]
            self.answer_tokens[item] = answer_tokens
        answer_mask = answer_tokens.ge(0)  # mask is zero where we out of sequence
        answer_tokens[~answer_mask] = 0
        answer_mask = answer_mask.float()

        return query_tokens, query_mask, answer_tokens, answer_mask

    def get_image(self, item):
        name = str(self.img_paths[item])
        # name = f"{self.img_path}/{name}"
        image_resized = Image.open(name)
        image_resized = image_resized.resize((256, 256))
        return image_resized
        # image_resized = cv2.resize(self.image_idx[item], (256,256))
        # return Image.fromarray(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, item):
        image = self.get_image(item)
        image = self.preprocess(image).unsqueeze(0)
        query_tokens, query_mask, answer_tokens, answer_mask = self.pad_tokens(item)
        return query_tokens, query_mask, answer_tokens, answer_mask, image[0], item
        # return query_tokens, query_mask, answer_tokens, answer_mask, item

    def show_image(self, item):
        image = self.get_image(item)
        text = self.tokenizer.decode(self.pad_tokens(item)[2])
        plt.imshow(image)
        print(text)

from accelerate import notebook_launcher
config = Config()
train_dataset = VQAv2_Dataset(config, dataset_path="/home/jovyan/vqa_project/baselines/VQAv2_train_translation.jsonl", imagespath_split="/home/jovyan/vqa_project/baselines/trainvqa/train2014/", coef_size=0.1)
val_dataset = VQAv2_Dataset(config, dataset_path="/home/jovyan/vqa_project/baselines/VQAv2_val_translation.jsonl", imagespath_split="/home/jovyan/vqa_project/baselines/valvqa/val2014/", coef_size=0.1)
args = ("bf16", 42, config, train_dataset, val_dataset)
# notebook_launcher(training_loop, args, num_processes=1)
training_loop(mixed_precision="fp16", args=config, train_dataset=train_dataset, val_dataset=val_dataset)