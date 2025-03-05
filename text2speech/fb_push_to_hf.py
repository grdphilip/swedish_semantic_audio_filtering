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
csv_path = './paths_dataset/updated_paths_dataset.csv'
audio_folder_path = './audio_files_wav/'

# Load your dataset
df = pd.read_csv(csv_path)

# Create a new DataFrame for the manifest
manifest = pd.DataFrame(columns=['audio', 'text', 'duration'])

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
manifest = []
for index, row in df.iterrows():
    for speaker_id in [f'audio_path_speaker_{i}' for i in range(1, 10)]:
        if pd.notna(row[speaker_id]):
            audio_path = os.path.join(audio_folder_path, os.path.basename(row[speaker_id]))
            audio_path = audio_path.split('.fla')[0] + '.wav'
            try:
                audio_data, duration, sr = load_audio(audio_path)
                # Convert audio data to binary format if needed, or use directly
                manifest.append({
                    'audio': {"path": audio_path, "array": audio_data, "sampling_rate": sr},  # Ensure path is correctly set
                    'text': row['sentence'],
                    'duration': duration,
                    'path': audio_path
                })
                
                
                
            except Exception as e:
                print(f"Failed to process audio {audio_path}: {str(e)}")
            break  # Process only the first non-empty speaker path

# Optionally save or process your manifest data further
with open('manifest.json', 'w') as f:
    json.dump(manifest, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


features = Features({
    "audio": Audio(sampling_rate=16000),
    "text": Value("string"),
    "duration": Value("float32"),
    "path": Value("string")
})

manifest_dict = {
    "audio": [item["audio"] for item in manifest],
    "text": [item["text"] for item in manifest],
    "duration": [item["duration"] for item in manifest],
    "path": [item["path"] for item in manifest]
}


dataset = Dataset.from_dict(manifest_dict, features=features)
print(dataset)
dataset.push_to_hub("grdphilip/facebook_m4t_v2_syndata")


    


