import asyncio
from pydub import AudioSegment
import os
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
import soundfile as sf
import pandas as pd
from glob import iglob
import re
import ast

load_dotenv()
api_key = os.getenv("PETTER_ELEVENLABS")
client = AsyncElevenLabs(api_key=api_key)

def resample_and_save_audio_sync(audio_stream, output_mp3_path, output_wav_path, target_sample_rate=16000):
    with open(output_mp3_path, "wb") as mp3_file:
        for chunk in audio_stream:
            mp3_file.write(chunk)
    audio = AudioSegment.from_file(output_mp3_path, format="mp3")
    resampled_audio = audio.set_frame_rate(target_sample_rate)
    resampled_audio.export(output_wav_path, format="wav")
    print(f"Resampled audio saved to {output_wav_path}")
    
async def text_to_audio_and_save(text, speaker_id, index, save_path="./elevenlabs_audio_files"):
    # 'convert' returns an async generator
    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=speaker_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    output_mp3_path = f"{save_path}/audio_speaker_{speaker_id}_{index}.mp3"
    output_wav_path = f"{save_path}_resampled/audio_speaker_{speaker_id}_{index}.wav"

    # Write chunks as they arrive
    with open(output_mp3_path, "wb") as mp3_file:
        async for chunk in audio_stream:
            mp3_file.write(chunk)

    # Now use pydub to resample (sync)
    audio = AudioSegment.from_file(output_mp3_path, format="mp3")
    resampled_audio = audio.set_frame_rate(16000)
    resampled_audio.export(output_wav_path, format="wav")
    print(f"Resampled audio saved to {output_wav_path}")

    return output_wav_path


def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as sound_file:
        sample_rate = sound_file.samplerate
        number_of_frames = sound_file.frames
        duration_seconds = number_of_frames / sample_rate
    return duration_seconds

async def generate_audio_async(
    speakers, target_hours_per_speaker, seconds_needed_per_speaker,
    generated_seconds_per_speaker, start_index_for_next_speaker, utterances
):
    print("Generating audio (async)...")

    for speaker_id in speakers:
        # Build a list of tasks for each speaker
        tasks = []
        row_indices = []
        for index, row in utterances.iloc[start_index_for_next_speaker:].iterrows():
            # Stop adding tasks if we have enough audio for this speaker
            if generated_seconds_per_speaker[speaker_id] >= seconds_needed_per_speaker:
                break
            tasks.append(
                text_to_audio_and_save(row["sentence"], speaker_id, index)
            )
            row_indices.append(index)

        # Run the tasks concurrently
        audio_paths = await asyncio.gather(*tasks)

        # Update durations and break if needed
        for audio_path, idx in zip(audio_paths, row_indices):
            duration_seconds = get_audio_duration(audio_path)
            utterances.at[idx, f'audio_path_speaker_{speaker_id}'] = audio_path
            generated_seconds_per_speaker[speaker_id] += duration_seconds
            if generated_seconds_per_speaker[speaker_id] >= seconds_needed_per_speaker:
                print(f"Completed generating {target_hours_per_speaker} hours for speaker {speaker_id}.")
                start_index_for_next_speaker = idx + 1
                break

    utterances.to_csv("./elevenlabs_paths_dataset/dataset_final.csv", index=False)
    return utterances.head()

def get_path_list(path="./elevenlabs_paths_dataset"):
    path_list = []
    for fn in iglob(f'{path}/*'):
        print(fn)
        path_list.append(fn)
    return path_list

def extract_audio_numbers(file_paths):
    pattern = r'audio_speaker[1367]_(\\d+)\\.flac'
    extracted_numbers = []
    for path in file_paths:
        match = re.search(pattern, path)
        if match:
            extracted_numbers.append(int(match.group(1)))
    return extracted_numbers

def find_disruption_pairs(sorted_numbers):
    disruptions = []
    for i in range(1, len(sorted_numbers)):
        if sorted_numbers[i] - sorted_numbers[i - 1] != 1:
            disruptions.append((sorted_numbers[i - 1], sorted_numbers[i]))
    return disruptions

def convert_str_to_dict(string):
    return ast.literal_eval(string)

# Example usage:
# asyncio.run(generate_audio_async([...], target_hours_per_speaker=1, seconds_needed_per_speaker=3600, ...))
