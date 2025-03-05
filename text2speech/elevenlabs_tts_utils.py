
from pydub import AudioSegment
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs    
import soundfile as sf
import pandas as pd
from glob import glob, iglob
import re
import ast


load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(
    api_key=api_key
)


def resample_and_save_audio(audio_stream, output_mp3_path, output_wav_path, target_sample_rate=16000):
    # Blir ganska mycket brusigare med resample till 16 kHz
    with open(output_mp3_path, "wb") as mp3_file:
        for chunk in audio_stream:
            mp3_file.write(chunk)
    
    audio = AudioSegment.from_file(output_mp3_path, format="mp3")
    resampled_audio = audio.set_frame_rate(target_sample_rate)
    resampled_audio.export(output_wav_path, format="wav")
    print(f"Resampled audio saved to {output_wav_path}")


def text_to_audio_and_save(text, speaker_id, index, save_path="./elevenlabs_audio_files"):
    
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=speaker_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    
    output_mp3_path = f"{save_path}/audio_speaker_{speaker_id}_{index}.mp3"
    output_wav_path = f"{save_path}_resampled/audio_speaker_{speaker_id}_{index}.wav"
    
    resample_and_save_audio(audio, output_mp3_path, output_wav_path)
    return output_wav_path
    
    
def get_audio_duration(file_path):
    # Open the file and retrieve its sample rate and number of frames
    with sf.SoundFile(file_path) as sound_file:
        sample_rate = sound_file.samplerate
        number_of_frames = sound_file.frames
        duration_seconds = number_of_frames / sample_rate
        print(duration_seconds)
    return duration_seconds
    

def generate_audio(speakers, target_hours_per_speaker, seconds_needed_per_speaker, generated_seconds_per_speaker, start_index_for_next_speaker, utterances):
    print("Generating audio...")
    
    for speaker_id in speakers:
        for index, row in utterances.iloc[start_index_for_next_speaker:].iterrows():
            if generated_seconds_per_speaker[speaker_id] < seconds_needed_per_speaker:
                audio_path = text_to_audio_and_save(row["sentence"], speaker_id, index)
                duration_seconds = get_audio_duration(audio_path)
                utterances.at[index, f'audio_path_speaker_{speaker_id}'] = audio_path
                generated_seconds_per_speaker[speaker_id] += duration_seconds
                
                if generated_seconds_per_speaker[speaker_id] >= seconds_needed_per_speaker:
                    print(f"Completed generating {target_hours_per_speaker} hours for speaker {speaker_id}.")
                    start_index_for_next_speaker = index + 1
                    break
            else:
                break
            
    utterances.to_csv("./elevenlabs_paths_dataset/dataset_final.csv", index=False)
    return utterances.head()

def get_path_list(path= "./elevenlabs_paths_dataset"):
    pathList= []
    for fn in iglob(pathname=f'{path}/*'):
        print(fn)
        pathList.append(fn)
    return pathList


def extract_audio_numbers(file_paths):
    # This pattern matches 'audio_speaker' followed by 1, 5, or 7, then an underscore and one or more digits (\d+),
    # capturing those digits before the '.flac' extension. It assumes these specific speaker numbers are of interest.
    pattern = r'audio_speaker[1367]_(\d+)\.flac'
    extracted_numbers = []

    for path in file_paths:
        # Search for the pattern in each path
        match = re.search(pattern, path)
        if match:
            # If a match is found, convert the captured group (the numbers) to an integer and add to the list
            extracted_numbers.append(int(match.group(1)))
    return extracted_numbers


def find_disruption_pairs(sorted_numbers):
    disruptions = []
    # Iterate through the sorted list, starting from the second item
    for i in range(1, len(sorted_numbers)):
        if sorted_numbers[i] - sorted_numbers[i - 1] != 1:
            # Add the number before the disruption and the disrupted value as a tuple
            disruptions.append((sorted_numbers[i - 1], sorted_numbers[i]))
    
    return disruptions

def convert_str_to_dict(string):
    return ast.literal_eval(string)