from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments

from tqdm.auto import tqdm

import wandb

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from torch.amp import autocast


def load_embs(dataset_path, tokenizer, dataset_size=1):
    embs = []
    count = 0

    train_dataset = load_dataset(dataset_path, split="train", streaming=True)

    for data in tqdm((train_dataset)):
        prompt = "<LM>" + data["system_prompt"] + " " + data["question"]
        response = data["response"]

        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, truncation=True, max_length=1024
        )
        output_ids = tokenizer.encode(response, add_special_tokens=False)

        if len(input_ids) < 768 and len(output_ids) < 768:
            embs.append({"input_ids": input_ids, "output_ids": output_ids})
            count += 1
        if count == dataset_size:
            break

    return embs


class OrcaDataset(Dataset):
    def __init__(self, tokenizer, embs):
        self._data = []
        self.tokenizer = tokenizer
        self.max_input_len = 0
        self.max_output_len = 0

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

        for emb in embs:
            input_ids = emb["input_ids"]
            output_ids = emb["output_ids"] + [self.eos_token_id]
            self._data.append((input_ids, output_ids))
            self.max_input_len = max(self.max_input_len, len(input_ids))
            self.max_output_len = max(self.max_output_len, len(output_ids))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int):
        input_ids, output_ids = self._data[item]

        input_npad = self.max_input_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * input_npad
        input_ids = input_ids + input_npad * [self.pad_token_id]

        output_npad = self.max_output_len - len(output_ids)
        labels = output_ids + output_npad * [-100]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": attention_mask,
            "labels": torch.LongTensor(labels),
        }


checkpoint = "ai-forever/FRED-T5-1.7B"
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, eos_token="</s>")
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

dataset_path = "d0rj/OpenOrca-ru"
dataset_size = 2_000_000
loaded_data = load_embs(dataset_path, tokenizer, dataset_size)

orca_dataset = OrcaDataset(tokenizer, loaded_data)

wandb.login(key="KEY", relogin=True)
wandb.init(sync_tensorboard=True, name="NAME", project="PROJECT", entity="ENTITY")

batch_size = 4

training_args = TrainingArguments(
    output_dir="./FRED-T5-tune",
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.05,
    optim="adamw_hf",
    lr_scheduler_type="linear",
    warmup_steps=1_000,
    report_to="wandb",
    run_name="train",
    gradient_accumulation_steps=10,
    # use_cpu=True
)

trainer = Trainer(
    model=model,
    train_dataset=orca_dataset,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
