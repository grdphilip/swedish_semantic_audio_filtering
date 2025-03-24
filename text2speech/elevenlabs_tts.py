import asyncio
from itertools import cycle
import os
import re
import ast
import pandas as pd
from pydub import AudioSegment
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
import soundfile as sf
from glob import iglob

load_dotenv()
api_key = os.getenv("PETTER_ELEVENLABS")
client = AsyncElevenLabs(api_key=api_key)

CONCURRENCY_LIMIT = 15
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

def get_audio_duration(file_path):
    with sf.SoundFile(file_path) as s:
        return s.frames / s.samplerate  # seconds

async def text_to_audio_and_save(text, speaker_id, index, save_path="./elevenlabs_audio_files"):
    async with semaphore:
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=speaker_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        output_mp3_path = f"{save_path}/audio_speaker_{speaker_id}_{index}.mp3"
        output_wav_path = f"{save_path}_resampled/audio_speaker_{speaker_id}_{index}.wav"

        # Write the MP3 chunks
        with open(output_mp3_path, "wb") as mp3_file:
            async for chunk in audio_stream:
                mp3_file.write(chunk)

        # Resample (synchronously) with pydub
        audio = AudioSegment.from_file(output_mp3_path, format="mp3")
        resampled_audio = audio.set_frame_rate(16000)
        resampled_audio.export(output_wav_path, format="wav")

        duration_s = get_audio_duration(output_wav_path)
        return output_wav_path, duration_s

async def generate_audio_by_sentences_async(utterances, speakers, batch_size=25):
    speaker_cycle = cycle(speakers)
    total_rows = len(utterances)
    total_duration_s = 0.0  # track total duration in seconds

    print(f"Starting round-robin TTS generation over {total_rows} rows...")

    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        chunk = utterances.iloc[start_idx:end_idx]

        tasks = []
        row_indices = []

        print(f"\nProcessing batch {start_idx} to {end_idx - 1}...")

        for index, row in chunk.iterrows():
            speaker_id = next(speaker_cycle)
            tasks.append(text_to_audio_and_save(row["sentence"], speaker_id, index))
            row_indices.append(index)

        # Gather results for this batch
        results = await asyncio.gather(*tasks)

        # Update DataFrame with paths & durations
        for (file_path, duration_s), idx in zip(results, row_indices):
            utterances.at[idx, 'audio_path'] = file_path
            utterances.at[idx, 'audio_duration_s'] = duration_s
            total_duration_s += duration_s

        # Save partial results
        utterances.to_csv("./elevenlabs_paths_dataset/dataset_final.csv", index=False)

        # Print how many minutes so far
        total_duration_min = total_duration_s / 60
        print(f"Saved partial results up to row {end_idx - 1}. "
              f"Total audio so far: {total_duration_min:.2f} minutes.")

    print("All batches processed.")
    return utterances

def get_path_list(path="./elevenlabs_audio_files"):
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

async def main():
    # Load the full dataset
    utterances = pd.read_csv("../text_generation/dataset/final_sentences.csv")

    # Define speakers
    speakers = [
        "aSLKtNoVBZlxQEMsnGL2",
        "7UMEOkIJdI4hjmR2SWNq",
        "fFe6F6cCl526GpIxiUxu",
        "XB0fDUnXU5powFXDhCwa",
        "pqHfZKP75CvOlQylNhV4"
    ]

    # Generate audio in batches of 25, printing partial durations each time
    await generate_audio_by_sentences_async(utterances, speakers, batch_size=25)

if __name__ == '__main__':
    asyncio.run(main())
