from huggingface_hub import Repository, login
from datasets import Dataset, load_dataset
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

# Load your dataset
df = pd.read_csv(csv_path)

# Create a new DataFrame for the manifest
manifest = pd.DataFrame(columns=['audio', 'text', 'duration', 'path'])

# Function to get duration of an audio file
def get_audio_duration(file_path):
    try:
        return librosa.get_duration(filename=file_path)
    except Exception:
        return None  # In case the file does not exist or other issues
    

def load_audio(audio_path):
    data, samplerate = sf.read(audio_path)
    duration = librosa.get_duration(y=data, sr=samplerate)
    return data, duration, samplerate

# Process each row and collect data
#speakers = ["x0u3EW21dbrORJzOq1m9", "4xkUqaR9MYOJHoaC1Nak", "kkwvaJeTPw4KK0sBdyvD"]

manifest = []
total_duration = 0.0

for index, row in df.iterrows():
    if pd.notna(row['audio_path']):
        audio_path = os.path.join(audio_folder_path, os.path.basename(row['audio_path']))
        audio_path = audio_path.split('.wav')[0] + '.mp3'
        print(audio_path)
        try:
            audio_data, duration, sr = load_audio(audio_path)
            total_duration += duration  # accumulate duration

            manifest.append({
                'audio': {
                    "array": audio_data,
                    "sampling_rate": sr,
                    'path': audio_path
                },
                'text': row['sentence'],
                'duration': duration,
                'path': audio_path,
                'entities': json.dumps(row['entity']),
                'metadata': json.dumps(row['entity_type'])
            })
        except Exception as e:
            print(f"Failed to process audio {audio_path}: {str(e)}")

print(f"Total duration of all audio: {total_duration} seconds")
print(f"Total duration in hours: {total_duration/3600:.2f} hrs")


# Optionally save or process your manifest data further
with open('manifest.json', 'w') as f:
    json.dump(manifest, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


features = Features({
    "audio": Audio(sampling_rate=16000),
    "text": Value("string"),
    "duration": Value("float32"),
    "path": Value("string"),
    "entities": Value("string"),
    "metadata": Value("string")
    
})

manifest_dict = {
    "audio": [item["audio"] for item in manifest],
    "text": [item["text"] for item in manifest],
    "duration": [item["duration"] for item in manifest],
    "path": [item["path"] for item in manifest],
    "entities": [item["entities"] for item in manifest],
    "metadata": [item["metadata"] for item in manifest]
    
}


dataset = Dataset.from_dict(manifest_dict, features=features)
print(dataset)
dataset.push_to_hub("grdphilip/elevenlabs_syndata")


    


