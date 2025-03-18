import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import librosa
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from jiwer import wer, cer
import re
import ast 

def create_dataloaders(filenames, batch_size, data_collator):
    dataloaders = []
    dataframes = []
    datasets = []

    for filename in filenames:
        # Read JSON file
        manifest_data = read_json_file(filename)
        # Create DataFrame
        df = pd.DataFrame(manifest_data)
        dataframes.append(df)
        # Create Dataset
        dataset = MyDataset(df)
        datasets.append(dataset)
        # Create DataLoader
        dataloader = CustomDataLoader(dataset=dataset, batch_size=batch_size, collate_fn=data_collator)
        dataloaders.append(dataloader)

    return dataloaders, dataframes


# Function to read a JSON file with potentially multiple JSON objects
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        # Read the entire file content
        file_content = file.read()
        
        # Split the content by lines and process each line separately
        json_objects = []
        for line in file_content.splitlines():
            if line.strip():  # Ignore empty lines
                try:
                    json_objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
    
    return json_objects

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, collate_fn=None):
        """
        Args:
            dataset: An instance of your custom dataset class.
            batch_size (int): Number of samples in each batch.
            collate_fn (callable, optional): A function to collate samples into batches.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = list(range(len(dataset)))
        self.collate_fn = collate_fn
        self.number_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for idx in range(self.number_batches):
            start_idx = idx * self.batch_size
            end_idx = min((idx + 1) * self.batch_size, len(self.dataset))  # Adjust end index to avoid out-of-bounds
            batch = self.dataset[start_idx:end_idx]
            if self.collate_fn:
                batch = self.collate_fn(batch)

            yield batch

    def __len__(self):
        """Returns the number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):

        file_paths = [sample for sample in features["input_features"]]

        input_features = []

        for file_path in file_paths:
            audio_data, sample_rate = librosa.load(file_path, sr=None, dtype=np.float32)
            if sample_rate != 16000: 
                print("Resampling audio")
                audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)

            input_features.append(audio_data)
        
        batch = input_features

        return batch
    

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
      
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx].copy()
        
        return {
            'input_features': data['audio_filepath']
        }
         # For safer literal evaluation

def clean_entities(raw_entities):
    cleaned_entities = []

    for entity_string in raw_entities:
        # Fix double escaping issues by replacing `\\"` with `"`, then strip leading/trailing quotes
        entity_string = entity_string.replace('\\"', '"').strip('"')

        # Convert JSON-like string to a Python list
        try:
            entity_list = json.loads(entity_string)
        except json.JSONDecodeError:
            continue  # Skip invalid entries

        # Decode Unicode escape sequences properly and split entities into words
        cleaned_group = [word for entity in entity_list for word in entity.encode().decode("unicode_escape").strip().split()]
        
        cleaned_entities.append(cleaned_group)  # Append the processed group

    return cleaned_entities

def calculate_entity_precision(normalized_cands, entities_ref):
    entities_total = len(entities_ref)
    correctly_identified_entities = 0
    
    missed_entities = []    

    for idx, val in enumerate(normalized_cands):
        for entity in entities_ref[idx]:
            if entity in val:
                correctly_identified_entities += 1
                break
            else:
                missed_entities.append({"entity": entity, "candidate": val})    
                
            
    if entities_total == 0:
        return 0.0
    
    return correctly_identified_entities / entities_total, missed_entities

    


def calculate_and_store_metrics(references, candidates, entities, transform_func, subset_name, results_df):
    """Calculate WER and CER, print and store the results in a DataFrame."""
    # Normalize the references and candidates
    normalized_refs = [' '.join(transform_func(ref)[0]) for ref in references]
    normalized_cands = [' '.join(transform_func(cand['text'])[0]) for cand in candidates]
    
    entity_score, missed_entities = calculate_entity_precision(normalized_cands, entities)
    print(len(missed_entities))

    # Calculate metrics
    wer_score = wer(normalized_refs, normalized_cands)
    cer_score = cer(normalized_refs, normalized_cands)
    print(f"entity_score {entity_score}")
    print(f"wer_score {wer_score}") 
    print(f"cer_score {cer_score}")

    # Print results
    print(subset_name)
    print("Word Error Rate (WER):", wer_score)
    print("Character Error Rate (CER):", cer_score)

    # Create a DataFrame with the results and append to the main results DataFrame
    df = pd.DataFrame({"SUBSET": [subset_name], "WER": [wer_score], "CER": [cer_score]})
    return pd.concat([results_df, df], ignore_index=True)