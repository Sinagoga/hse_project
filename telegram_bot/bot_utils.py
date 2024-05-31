import torch
import pickle
import open_clip

model = pickle.load(open("model_path.pkl", 'rb'))
clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-plus-240", pretrained="laion400m_e32")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

torch.cuda.empty_cache()
torch.manual_seed(42)

print("Model loaded.")

def bot_pref(image, question):
    image = image.resize((224, 224))
    image = torch.tensor(preprocess(image).unsqueeze(0))
    out = model.generate(image, [question], max_seq_len=40)
    return out
