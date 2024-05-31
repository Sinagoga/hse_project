import torch
import math
import torch.nn.functional as F
import yaml

def decode_question(question_token, tokenizer):
    decoded_string = tokenizer.decode(question_token)
    decoded_string = decoded_string.replace("<pad>", "")
    return decoded_string

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True)
    return t.softmax(dim=dim)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            self.__dict__[key] = value

def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return Config(config_data)
