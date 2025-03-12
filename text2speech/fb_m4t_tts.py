import pandas as pd 
import re
from fb_m4t_tts_utils import convert_str_to_dict,gen_audio, get_path_list, extract_audio_numbers, find_disruption_pairs
import os

#Original capes dataset https://huggingface.co/datasets/soarescmsa/capes/viewer/en-pt/train?p=1
capes = pd.read_csv("../text_generation/dataset/utterances.csv")
fbm2tts = True


print("Cleaning the dataset...")
print(capes)

#capes['sentence'] = capes['sentence'].apply(convert_str_to_dict)



#pattern = re.compile(r'[\[\]\{\}\(\)\\/&%#\*\+_<>\"]')

#capes = capes[~capes['sentence'].apply(lambda x: pattern.search(x) is not None)]


# The speakers to generate audio
speakers =  [1,2,3,4,5,6,7,8,9]

# The target number of hours per speaker
target_hours_per_speaker = 0

# Convert hours to seconds
#seconds_needed_per_speaker = target_hours_per_speaker * 3600  # Convert hours to seconds
seconds_needed_per_speaker = 10
generated_seconds_per_speaker = {speaker: 0 for speaker in speakers}

# Initialize a variable to keep track of the starting index for the next speaker
start_index_for_next_speaker = 0

print(f"Generating {target_hours_per_speaker} hours of audio for each speaker...")
print("This may take a while...")

#Generate the audio
if fbm2tts: gen_audio(speakers=speakers, target_hours_per_speaker=target_hours_per_speaker, seconds_needed_per_speaker=seconds_needed_per_speaker, generated_seconds_per_speaker=generated_seconds_per_speaker, start_index_for_next_speaker=start_index_for_next_speaker, capes=capes)


path_LIST= get_path_list(path= "./dataset/generatedSynAudios")
numbers = extract_audio_numbers(path_LIST)
disruption_pairs = find_disruption_pairs(sorted(numbers)) 


index_path_map = {int(path.split('_')[-1].split('.')[0]): path for path in path_LIST}
capes['Audio_path'] = capes.index.map(index_path_map.get)
capes_final= capes.dropna(subset=["Audio_path"])
save_path_audio_paths = "./paths_dataset"
if not os.path.exists(save_path_audio_paths):
    os.makedirs(save_path_audio_paths)
    
capes_final.to_csv("./paths_dataset/dataset_final.csv", index=False)

############ AT THE END A CSV FILE WITH THE PATH TO GENERATED AUDIO FILES IS CREATED ############re

"""
==============================================================================================================

Venv grejer jag alltid glömmer bort 
python3.10 -m venv venv
source venv/bin/activate
pip3.10 install -r requirements.txt
deactivate

==============================================================================================================

Resurser för data:
https://github.com/gongouveia/Whisper-Synthetic-ASR-Dataset-Generator
https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html

- Lots of samples
- Lots of different voices
- Same voice with different pronunciations
- Lots of negative samples (Words that are not keyword(s)) but sound similar or commonly heard in the field


==============================================================================================================
ENGINES:

https://elevenlabs.io/text-to-speech API
https://huggingface.co/facebook/seamless-m4t-v2-large OPEN-SOURCE

==============================================================================================================

# Frågor att ta med till fredag:
Vill man ha flera talare för samma mening? 

==============================================================================================================
"""