from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


model = BlipForQuestionAnswering.from_pretrained("C:/Users/lozhn/Downloads/Telegram Desktop/model_10epoch")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)

print("Model loaded.")

def bot_pref(path2img, text):
    image = Image.open(path2img).convert("RGB")
    question = text
    
    inputs = processor(image, question, padding="max_length", truncation=True, return_tensors="pt",  max_length=60).to("cuda:0")
    out = model.generate(**inputs)
    
    return processor.batch_decode(out, skip_special_tokens=True)