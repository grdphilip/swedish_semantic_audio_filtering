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


df = pd.read_csv('results/finan_sentences.csv')