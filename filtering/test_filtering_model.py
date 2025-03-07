# 1. Get good and bad data from huggingface
# 2. Merge into one dataset
# 3. Add a column for similarity score
# 4. Check the similarity score for the data
# 5. Heuristically evaluate the threshold for the similarity score

from model import MusCALL
import torch
from transformers import AutoTokenizer
import librosa


def get_audio(self, idx):
        
    audio_data, sample_rate = librosa.load(f".{self.audio_paths[idx]}", sr=None, dtype=np.float32)
    if sample_rate != 16000: 
        print("Resampling audio")
        print("Sample id: ", self.audio_paths[idx])
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
    return audio_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "./save/experiments/model1/checkpoint.pth.tar"
checkpoint = torch.load(checkpoint_path, map_location=device)

model = MusCALL()

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("KBLab/bert-base-swedish-cased")

text = ""
text_tokens = tokenizer(
    text, return_tensors="pt", padding=True, truncation=True, max_length=512
)

audio_path = ""

waveform = get_audio(audio_path)
waveform = waveform.to(device)

with torch.no_grad():
    print(f"Waveform shape {waveform.shape}, text shape {text_tokens['input_ids'].shape}")
    audio_embedding = model.encode_audio(waveform)
    text_embedding = model.encode_text(text_tokens["input_ids"], text_tokens["attention_mask"])
    print(audio_embedding.shape, text_embedding.shape)
    similarity_score = torch.nn.functional.cosine_similarity(audio_embedding, text_embedding, dim=1)
    print(similarity_score)
    print(similarity_score.item())
