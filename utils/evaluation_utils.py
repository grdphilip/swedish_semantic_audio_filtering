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
        # Remove extra escaping, and strip any leading/trailing quotes
        entity_string = entity_string.replace('\\"', '"').strip('"')

        # Convert JSON-like string to a Python list
        try:
            entity_list = json.loads(entity_string)
        except json.JSONDecodeError:
            continue  # Skip invalid entries

        cleaned_group = []
        for entity in entity_list:
            # We directly decode the entity from the loaded JSON data to handle unicode escapes correctly
            decoded_entity = entity.encode('utf-8').decode('utf-8')
            cleaned_group.extend(decoded_entity.strip()) 

        cleaned_entities.append(cleaned_group)  # Append the processed group

    return cleaned_entities


def calculate_total_entities(entities_ref):
    total_entities = 0
    for entities in entities_ref:
        for entity in entities:
            total_entities += 1
    return total_entities

def calculate_entity_precision(normalized_cands, entities_ref, normalized_refs, metadata, normalized):
    # Kan det finnas någon mening med att kolla CER inne i entiteten
    # Exempel: Jwan - Jovan / Jwan - Jowan, Jakob - Jacob
    entities_total = calculate_total_entities(entities_ref)
    correctly_identified_entities = 0
    print(entities_total)   
    
    missed_entities = []    

    for idx, val in enumerate(normalized_cands):
        for i, entity in enumerate(entities_ref[idx]):
            if normalized:
                entity = entity.lower()
            if entity in val:
                correctly_identified_entities += 1
            else:
                missed_entities.append({"entity": entity, "candidate": val, "ground_truth": normalized_refs[idx], "type": metadata[idx]['entity_type'][i]})    
                
            
    if entities_total == 0:
        return 0.0
    
    return (entities_total - len(missed_entities)) / entities_total, missed_entities
    

    
def calculate_and_store_metrics(references, candidates, entities, metadata, transform_func, subset_name, results_df, normalized):
    """Calculate WER and CER, print and store the results in a DataFrame."""
    # Normalize the references and candidates
    normalized_refs = [' '.join(transform_func(ref)[0]) for ref in references]
    normalized_cands = [' '.join(transform_func(cand['text'])[0]) for cand in candidates]
    
    entity_score, missed_entities = calculate_entity_precision(normalized_cands, entities, normalized_refs, metadata, normalized)
    print(len(missed_entities))
    print(missed_entities)

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
    df = pd.DataFrame({"SUBSET": [subset_name], "WER": [wer_score], "CER": [cer_score], "ENTITY_ACCURACY": [entity_score]})
    missed_entities_df = pd.DataFrame(missed_entities)
    return pd.concat([results_df, df], ignore_index=True), missed_entities_df

