from datasets import Dataset, load_dataset
from huggingface_hub import list_datasets, login
from dotenv import load_dotenv
import logging
import os
import json
import openai
from openai import OpenAI, AsyncOpenAI
import re
from datasets import Audio, Features, Value, Sequence
import asyncio 
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_not_exception_type
from typing import List, Dict
from pydantic import BaseModel
from time import time



NUM_MAX_RETRY_ATTEMPTS = 2
MAX_COMPLETION_TOKENS = 3_000

load_dotenv()
# token = os.getenv('HF_WRITE')
# os.environ['HF_TOKEN'] = token
# login(token=token)

# datasets = list_datasets()
# print([d.id for d in datasets if "grdphilip" in d.id])

dataset = "fleurs"
update_path = "fleurs_corrected"

dataset_map: dict = {
    "cv": "grdphilip/cv_swedish_with_entities",
    "fleurs": "grdphilip/fleurs_swedish_with_entities",
    "cv_corrected": "grdphilip/cv_swedish_with_entities_corrected",
    "fleurs_corrected": "grdphilip/fleurs_swedish_with_entities_corrected",
}

df = load_dataset(dataset_map[dataset])['train']


logging.basicConfig(filename='entity_correction.log', level=logging.INFO, format='%(asctime)s %(message)s')
api_key = os.getenv("TOMAS_OPENAI_KEY")
client = OpenAI(api_key=api_key)


def correct_extracted_entities(df: Dataset) -> Dataset:
    prompt_template = (
        "You are given a sentence (text), extracted entities, and their metadata. "
        "Your task is ONLY to fix obvious formatting errors in entities and metadata:\n"
        "- Merge entities mistakenly split.\n"
        "- Separate entities mistakenly joined.\n"
        "- Ensure entities EXACTLY match substrings from the original text.\n"
        "- DO NOT add additional information from the text about an entity that is not already present in the entity.\n"
        "- ALL text in the reformatted entity MUST be present in the original entity.\n"
        "- ALL entities must EXACTLY match the substrings as they appear in the original text, even if grammatically incorrect.\n"
        "\n"
        "Return ONLY corrected entities and metadata in plain JSON:\n"
        "{{\"entities\": [\"entity1\", \"entity2\"], \"metadata\": {{\"entity\": [\"entity1\", \"entity2\"], \"entity_type\": [\"type1\", \"type2\"]}}}}\n\n"
        "Text: {text}\n"
        "Entities: {entities}\n"
        "Metadata: {metadata}\n\n"
        "Corrected JSON:"
    )

    corrected_rows = []

    for i, row in enumerate(df.select(range(50))):
        prompt = prompt_template.format(
            text=row['text'],
            entities=row['entities'],
            metadata=row['metadata']
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = re.sub(r'^```json|```$', '', response.choices[0].message.content).strip()
        corrected_json = json.loads(content)

        print(f"Row {i}: Original Entities: {row['entities']} -> Corrected Entities: {corrected_json['entities']}")

        # Update the row
        row['entities'] = corrected_json['entities']
        row['metadata'] = corrected_json['metadata']

        corrected_rows.append(row)

    corrected_df = Dataset.from_dict({k: [row[k] for row in corrected_rows] for k in corrected_rows[0]})
    
    return corrected_df

    

# features = Features({
#     'audio': Audio(sampling_rate=16000),
#     'path': Value('string'),
#     'text': Value('string'),
#     'entities': Sequence(Value('string')),
#     'metadata': Sequence({
#         'entity': Value('string'),
#         'entity_type': Value('string')
#     })
# })


# manifest_dict = {
#     'audio': [row['audio'] for row in df_corrected],
#     'path': [row['path'] for row in df_corrected],
#     'text': [row['text'] for row in df_corrected],
#     'entities': [row['entities'] for row in df_corrected],
#     'metadata': [row['metadata'] for row in df_corrected]
# }

# corrected_dataset = Dataset.from_dict(manifest_dict, features=features)
# corrected_dataset.push_to_hub(dataset_map[update_path])
# print(corrected_dataset)
# print(f"Pushed to {dataset_map[update_path]}")





