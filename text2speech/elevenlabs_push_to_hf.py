from huggingface_hub import Repository, login
from datasets import Dataset, load_dataset, DatasetDict
from dotenv import load_dotenv
import os
import pandas as pd
import librosa
import json
import soundfile as sf
import numpy as np
from datasets import Audio, Features, Value

# Login to Huggingface (you need to run `huggingface-cli login` beforehand)
# https://huggingface.co/datasets/my-north-ai/cv_mls_psfb_fs17_68k

load_dotenv()
token = os.getenv('HF_WRITE')
print(token)

# Reset the current active token
os.environ['HF_TOKEN'] = token

# Login with the new token
login(token=token)


# Define the path to your datasets
csv_path = './elevenlabs_paths_dataset/dataset_final.csv'
audio_folder_path = './elevenlabs_audio_files/'

df = pd.read_csv(csv_path)

def get_audio_duration(file_path):
    try:
        return librosa.get_duration(filename=file_path)
    except Exception:
        return None

# Build our manifest with lazy loading: only store the *file path* in "audio".
manifest = []
total_duration = 0.0

for index, row in df.iterrows():
    if pd.notna(row['audio_path']):
        # Adjust the path if you're using .mp3
        audio_path = os.path.join(audio_folder_path, os.path.basename(row['audio_path']))
        audio_path = audio_path.split('.wav')[0] + '.mp3'

        print(f"Processing: {audio_path}")
        duration = get_audio_duration(audio_path)
        if duration is None:
            print(f"Warning: could not determine duration for {audio_path}")
            continue

        total_duration += duration

        # Store only the path in "audio" for lazy loading
        manifest.append({
            "audio": audio_path,
            "text": row["sentence"],
            "duration": duration,
            "path": audio_path,
            "entities": json.dumps(row["entity"]),
            "metadata": json.dumps(row["entity_type"])
        })

print(f"\nTotal duration of all audio: {total_duration:.2f} seconds")
print(f"Total duration in hours: {total_duration / 3600:.2f} hrs")

# Optionally save the manifest
with open("manifest.json", "w") as f:
    json.dump(manifest, f)

# Define dataset features with Audio(...) for lazy loading
features = Features({
    "audio": Audio(sampling_rate=16000),  # Lazy load from file path
    "text": Value("string"),
    "duration": Value("float32"),
    "path": Value("string"),
    "entities": Value("string"),
    "metadata": Value("string")
})

# Create a single Dataset from the entire manifest
manifest_dict = {
    "audio": [item["audio"] for item in manifest],
    "text": [item["text"] for item in manifest],
    "duration": [item["duration"] for item in manifest],
    "path": [item["path"] for item in manifest],
    "entities": [item["entities"] for item in manifest],
    "metadata": [item["metadata"] for item in manifest],
}

full_dataset = Dataset.from_dict(manifest_dict, features=features)

# Shuffle and split into train/validation
full_dataset = full_dataset.shuffle(seed=42)
train_size = int(0.8 * len(full_dataset))  # 80% train, 20% validation
train_dataset = full_dataset.select(range(train_size))
val_dataset = full_dataset.select(range(train_size, len(full_dataset)))

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

print(dataset_dict)

# Push to your Hugging Face Hub repo with both splits
dataset_dict.push_to_hub("grdphilip/elevenlabs_syndata")