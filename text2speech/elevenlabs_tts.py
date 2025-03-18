from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import os
from pprint import pprint
import pandas as pd
from elevenlabs_tts_utils import generate_audio, get_path_list, extract_audio_numbers, find_disruption_pairs
from pydub import AudioSegment



utterances = pd.read_csv("../text_generation/dataset/utterances.csv")

# client = ElevenLabs(
#   api_key=api_key
# )

# audio = client.text_to_speech.convert(
#     text="Sju skönsjungande sjuksköterskor skötte sjuttiosju sjösjuka sjömän på skeppet Shanghai",
#     voice_id="4xkUqaR9MYOJHoaC1Nak",
#     model_id="eleven_multilingual_v2",
#     output_format="mp3_44100_128",
# )

# output_mp3_path = "output.mp3"
# output_wav_path = "resampled_output.wav"
# save_and_resample_audio(audio, output_mp3_path, output_wav_path)

speakers = ["x0u3EW21dbrORJzOq1m9", "4xkUqaR9MYOJHoaC1Nak", "kkwvaJeTPw4KK0sBdyvD"]
target_hours_per_speaker = 0
seconds_needed_per_speaker = 180
generated_seconds_per_speaker = {speaker: 0 for speaker in speakers}
start_index_for_next_speaker = 0

print(f"Generating {target_hours_per_speaker} hours of audio for each speaker...")
print("This may take a while...")


generate_audio(speakers = speakers, target_hours_per_speaker = target_hours_per_speaker, seconds_needed_per_speaker = seconds_needed_per_speaker, generated_seconds_per_speaker = generated_seconds_per_speaker, start_index_for_next_speaker = start_index_for_next_speaker, utterances=utterances)
path_list = get_path_list(path = "./elevenlabs_audio_files")
numbers = extract_audio_numbers(path_list)
disruption_pairs = find_disruption_pairs(sorted(numbers))

index_path_map = {int(path.split('_')[-1].split('.')[0]): path for path in path_list}
utterances['audio_path'] = utterances.index.map(index_path_map.get)
utterances_final = utterances.dropna(subset=["audio_path"])
utterances_final.to_csv("./elevenlabs_paths_dataset/dataset_final.csv", index=False)


#response = client.voices.get_all()
#voices_with_ids = [{"name": voice.name, "id": voice.voice_id, "labels": voice.labels} for voice in response.voices]
#print(len(voices_with_ids))
#pprint(voices_with_ids)


# Sjuka röster
# https://elevenlabs.io/app/voice-library filtrera på svenska
# https://elevenlabs.io/docs/api-reference/voices/get-all
# Adam composer: x0u3EW21dbrORJzOq1m9 - Funkar
# Sanna Hartfield conversational: 4xkUqaR9MYOJHoaC1Nak - Funkar 
# J Bengt: kkwvaJeTPw4KK0sBdyvD - Funkar

# Stöttat output format
# import typing

# OutputFormat = typing.Union[
#     typing.Literal[
#         "mp3_22050_32",
#         "mp3_44100_32",
#         "mp3_44100_64",
#         "mp3_44100_96",
#         "mp3_44100_128",
#         "mp3_44100_192",
#         "pcm_16000", ?
#         "pcm_22050",
#         "pcm_24000",
#         "pcm_44100",
#         "ulaw_8000",
#     ],
#     typing.Any,
# ]
# Kostnad: 500K credits ~ 10h audio 
# In the paper the generated 48972 hours ~ 7m credits 