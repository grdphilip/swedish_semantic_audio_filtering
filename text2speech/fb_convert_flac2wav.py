import os
from pydub import AudioSegment
import simpleaudio as sa

# Convert FLAC to WAV
def convert_flac_to_wav(flac_file_path, wav_file_path):
    audio = AudioSegment.from_file(flac_file_path, "flac")
    audio.export(wav_file_path, format="wav")

# Convert all FLAC files in a directory to WAV
def convert_all_flac_to_wav(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".flac"):
            flac_file_path = os.path.join(input_dir, file_name)
            wav_file_name = file_name.replace(".flac", ".wav")
            wav_file_path = os.path.join(output_dir, wav_file_name)
            convert_flac_to_wav(flac_file_path, wav_file_path)
            print(f"Converted {flac_file_path} to {wav_file_path}")

# Example usage
input_directory = "./audio_files"
output_directory = "./audio_files_wav"
convert_all_flac_to_wav(input_directory, output_directory)

# def play_audio(file_path):
#     wave_obj = sa.WaveObject.from_wave_file(file_path)
#     play_obj = wave_obj.play()
#     play_obj.wait_done()  # Wait until audio file is completely played

# # Play a WAV file (example)
# example_wav_file = os.path.join(output_directory, "audio_speaker7_15.wav")
# play_audio(example_wav_file)
