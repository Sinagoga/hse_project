class Config:
    encoder: str = "ViT-B-16-plus-240"
    llm: str = "ai-forever/FRED-T5-large"
    batch_size: int = 172
    num_epochs: int = 40
    frozen_gpt: int = 8
    frozen_clip: int = 24
    learning_rate: float  = 2e-4
    save_path: str = ""
    prefix_length: int = 60
    only_prefix: int = False
    prefix: str = "prefix_small"
    device: str = "cuda:0"
    save_every: int = 1
    warmup_steps: int = 2000