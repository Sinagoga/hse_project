from torch.utils.data import Dataset
import json
from PIL import Image
class VQAv2_Dataset(Dataset):
    def __init__(self, config, dataset_path,
                 tokenizer_name="", prefix_length=60, imagespath_split=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(config.encoder, pretrained="laion400m_e32")
        with open(dataset_path, 'r') as f:
            self.dataset = json.loads(list(f)[0])
        self.max_seq_len = prefix_length
        self.imagespath_split = imagespath_split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        if ("val" in self.imagespath_split):
            image_path = self.imagespath_split + self.dataset[idx]['image_id'].replace("train", "val") + ".jpg"
        else:
            image_path = self.imagespath_split + self.dataset[idx]['image_id'] + ".jpg"
        
        input_tokens = self.tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt")
        answer_tokens = self.tokenizer(
                answer,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len-20,
                return_tensors="pt")
                
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image = self.preprocess(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = clip_model.encode_image(image)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
        return input_tokens['input_ids'], input_tokens['attention_mask'], answer_tokens['input_ids'], answer_tokens['attention_mask'], image_features, idx