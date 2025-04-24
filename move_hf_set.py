from dotenv import load_dotenv
import os
from datasets import Dataset, load_dataset, DatasetDict

load_dotenv()
token = os.getenv('HF_WRITE')
print(token)

df = load_dataset("TonarTechnologies/elevenlabs_syndata_philip", "default",  token=token)

new_token = os.getenv('TONAR_HF_WRITE')


df.push_to_hub("TonarTechnologies/elevenlabs_syndata_philip", token=new_token)
