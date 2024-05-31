from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

model = BlipForQuestionAnswering.from_pretrained("model_path")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

torch.cuda.empty_cache()
torch.manual_seed(42)

print("Model loaded.")

def bot_pref(image, question):
    inputs = processor(image, question, padding="max_length", truncation=True, return_tensors="pt",  max_length=60).to(device)
    out = model.generate(**inputs)
    
    return processor.batch_decode(out, skip_special_tokens=True)
