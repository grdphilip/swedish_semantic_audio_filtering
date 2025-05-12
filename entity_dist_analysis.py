import pandas as pd
import numpy as np
import os
import sys
import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from jiwer import Compose, RemoveEmptyStrings, ToLowerCase, RemoveMultipleSpaces, Strip, RemovePunctuation, ReduceToListOfListOfWords
from utils.evaluation_utils import create_dataloaders, DataCollatorSpeechSeq2SeqWithPadding, calculate_and_store_metrics, clean_entities
from tqdm import tqdm
import pandas as pd
import os 

# 1. Benchmark hur distrubutionen ser ut i utvärderingsset
# 2. Benchmark hur distrubutionen ser ut efter fine-tuning


filenames = ["data/manifest_data/finetuning/raw/fleurs_swedish_with_entities_eval_manifest.json"]
#filenames = ["data/manifest_data/finetuning/raw/cv_swedish_with_entities_eval_manifest.json"]
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_path = "KBLab/kb-whisper-medium"    
    
prints = ["TEST"]
batch_size = 32
base_model = "KBLab/kb-whisper-medium"
processor = AutoProcessor.from_pretrained(base_model)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path,  # Instead of `pretrained_model`, use the checkpoint directory
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)

# Create dataloaders and corresponding dataframes
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
dataloaders, dataframes = create_dataloaders(filenames, batch_size, data_collator)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    batch_size=batch_size,
    torch_dtype=torch_dtype,
    device=device,
    chunk_length_s=30,
    generate_kwargs={"task": "transcribe", "language": "swedish"}
)

for dataloader, dataframe, subset_name in zip(dataloaders, dataframes, prints):
    # Collect candidates from the dataloader
    candidates = []
    for batch in tqdm(dataloader, desc=f"Processing {subset_name} audio files"):
        candidates.extend(pipe(batch))

    # Get reference texts
    references = dataframe['text'].to_list()
    reference_entities = dataframe['entities'].to_list()
    metadata = dataframe['metadata'].to_list()
    
    print(reference_entities)
    entities = clean_entities(reference_entities)
    print(references)
    print(entities)
    print(metadata)