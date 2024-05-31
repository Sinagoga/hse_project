import torch
import torch.nn as nn
from torch.nn import functional as nnf

from transformers.optimization import Adafactor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict
import pickle
from torchmetrics.text import BLEUScore
from statistics import mean
import wandb
from src.utils.utils import *
from src.utils.utils import load_config
from model import BILIP
from VQA_dataset import VQAv2_Dataset

bleu_scorers = [BLEUScore(n_gram=i) for i in [1, 2, 3]]

wandb.login(key="")
wandb.init(project="", sync_tensorboard=True, name="")


def train(model: nn.Module,
          optimizer: Any,
          scheduler: function,
          loss_func: nn.Module,
          loader: DataLoader,
          epoch: int,
          args: Dict[str, Any]):
    model.train()
    pbar = tqdm(loader, total=len(loader))
    step = 0
    for (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in pbar:

        query_tokens, query_mask, prefix = query_tokens.to(args.device), query_mask.to(args.device), prefix.to(
            args.device, dtype=torch.bfloat16)
        answer_tokens, answer_mask = answer_tokens.to(args.device), answer_mask.to(args.device)
        outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
        logits = outputs.logits
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                 ignore_index=0)

        loss2 = model.dist_loss(model.get_text_embeddings(answer_tokens).to(torch.float32), proj.to(torch.float32))
        loss += loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        #backpropogation
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_postfix({"loss": loss.item(), "dist_loss": loss2.item()})
        wandb.log({"loss": loss.item(), "dist_loss": loss2.item()})
        step += 1
        if step % 1000 == 0:
            print("QUESTION:", decode_question(query_tokens[0], train_dataset.tokenizer))
            print("ANSWER:", decode_question(answer_tokens[0], train_dataset.tokenizer))
            print("PREDICTED: ", model.generate(torch.tensor([train_dataset[idx[0]][4].tolist()]).to(args.device),
                                                [decode_question(query_tokens[0], model.tokenizer)], train_dataset.max_seq_len)[0])
    with open(f'{args.save_path}checkpoint_{epoch}.pkl', 'wb') as f:
        pickle.dump(model, f)

@torch.no_grad()
def evaluate(model: nn.Module,
             optimizer: Any,
             scheduler: Any,
             loss_func: nn.Module,
             loader: DataLoader,
             args: Dict[str, Any]):
    model.eval()
    pbar = tqdm(loader, total=len(loader))
    step = 0

    bl1 = []
    bl2 = []
    bl3 = []
    val_losses = []
    val_dist = []
    for (query_tokens, query_mask, answer_tokens, answer_mask, prefix, idx) in pbar:
        query_tokens, query_mask, prefix = query_tokens.to(args.device), query_mask.to(args.device), prefix.to(
            args.device, dtype=torch.bfloat16)
        answer_tokens, answer_mask = answer_tokens.to(args.device), answer_mask.to(args.device)
        outputs, proj = model(query_tokens, query_mask, answer_tokens, answer_mask, prefix)
        logits = outputs.logits
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), answer_tokens.flatten().to(torch.int64),
                                 ignore_index=0)
        loss2 = model.dist_loss(model.gpt.transformer.wte(answer_tokens), proj)

        real = [decode_question(answer_tokens[i], model.tokenizer) for i in range(len(answer_tokens))]
        pred = model.generate(torch.tensor([val_dataset[idx[j]][4].tolist() for j in range(len(idx))]).to(args.device),
                              [decode_question(query_tokens[j], model.tokenizer) for j in range(len(idx))], val_dataset.max_seq_len)
        
        bl1.append(bleu_scorers[0](pred, real))
        bl2.append(bleu_scorers[1](pred, real))
        bl3.append(bleu_scorers[2](pred, real))

        if step % 400 == 0:
            print("QUESTION:", decode_question(query_tokens[0], val_dataset.tokenizer))
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

        step += 1

        pbar.set_postfix({"val_loss": loss.item(), "val_dist": loss2.item()})
        val_losses.append(loss.item())
        val_dist.append(loss2.item())

    wandb.log({"val_loss": mean(val_losses),
               "val_dist": mean(val_dist)})
    wandb.log({
        "bleu_1": mean([tensor.item() for tensor in bl1]),
        "bleu_2": mean([tensor.item() for tensor in bl2]),
        "bleu_3": mean([tensor.item() for tensor in bl3]),
    })


def fit_model(args, model, train_loader, val_loader):
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size
    }

    model = model.to(args.device)

    wandb.watch(model, log_freq=10, log="gradients")

    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adafactor(model.parameters(), lr=args.learning_rate,
                          relative_step=False 
                          )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15000
    )
    evaluate(model, optimizer, scheduler, loss_func, val_loader, args)
    print("Start train model")
    for epoch in range(args.num_epochs):
        if epoch == args.frozen_gpt:
            print("GPT UNFROZEN")
            for p in model.gpt.parameters():
                p.requires_grad = True
        if epoch == args.frozen_clip:
            print("CLIP UNFROZEN")
            for p in model.clip_model.parameters():
                p.requires_grad = True
        print(f"---------- Train epoch {epoch} ---------")
        train(model, optimizer, scheduler, loss_func, train_loader, epoch, args)
        print(f"---------- Evaluate epoch {epoch} ---------")
        evaluate(model, optimizer, scheduler, loss_func, val_loader, args)

config = load_config("/Users/ildarkhamaganov/hse_project_vqa/hse_project/config.yaml")
train_dataset = VQAv2_Dataset(config, dataset_path="VQAv2_train_translation.jsonl", imagespath_split="trainvqa/train2014/", coef_size=0.5)
val_dataset = VQAv2_Dataset(config, dataset_path="VQAv2_val_translation.jsonl", imagespath_split="valvqa/val2014/", coef_size=0.05)

model = BILIP(config, config.prefix_length)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=20, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=20, shuffle=True, drop_last=False)

fit_model(config, model, train_loader, val_loader)