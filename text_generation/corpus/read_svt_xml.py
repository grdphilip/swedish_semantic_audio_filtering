
import xml.etree.ElementTree as ET
import csv
from transformers import pipeline
import re
import json

# Load Swedish BERT NER model
nlp = pipeline("ner", model="KBLab/bert-base-swedish-cased-ner", tokenizer="KBLab/bert-base-swedish-cased-ner")

# Define input/output files
xml_file = "svt-2023.xml"
csv_file = "sentences_entities_bert.csv"

# Helper function to extract capitalized words as potential names
def extract_capitalized_words(sentence, found_entities):
    words = sentence.split()
    possible_names = []
    for i, word in enumerate(words):
        # If the word starts with an uppercase letter and is not the first word (to avoid sentence-start capitalization)
        if re.match(r"^[A-ZÅÄÖ][a-zåäö]+$", word) and i > 0:
            if word not in found_entities and len(word) > 3:
                print(f"Found possible name: {word}")
                possible_names.append(word)
    return possible_names

def clean_sentence(sentence: str) -> str:
    # Remove leading '-'
    sentence = re.sub(r'^-\s*', '', sentence)
    
    # Replace '-' with ','
    sentence = sentence.replace('–', ',')
    
    sentence = sentence.replace('”', '')
    
    # Remove all commas
    sentence = sentence.replace(',', '')
    
    # Replace ' : ' with ' ', but not when words are adjacent (e.g., SAKO:s remains unchanged)
    sentence = re.sub(r'(?<=\s):\s', ' ', sentence)
    
    # Remove occurrences of [ ... ]
    sentence = re.sub(r'\[.*?\]', '', sentence)
    
    # Replace middle quotation marks with a comma before the quoted word
    sentence = re.sub(r'(?<=\S)\s*["””]\s*(?=\S)', ',', sentence)
    
    return sentence

# Open CSV file for writing
with open(csv_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sentence", "entities", "metadata"])  # CSV headers

    # Parse XML
    context = ET.iterparse(xml_file, events=("start", "end"))
    count = 0

    for event, elem in context:
        if event == "end" and elem.tag == "sentence":
            # Extract sentence text
            words = [token.text for token in elem.findall(".//token") if token.text]
            sentence = " ".join(words)
            sentence = clean_sentence(sentence)
        
        
            # Run BERT NER model
            bert_entities = []
            current_entity = ""
            current_label = None
            previous_end = -1  # Tracks end position of the last processed token

            for token in nlp(sentence):
                word = token["word"]
                label = token["entity"]
                score = token["score"]
                start = token["start"]
                end = token["end"]

                if word.startswith("##"):
                    # Merge with previous token without space
                    current_entity += word[2:]
                else:
                    # If this token continues an entity (same label, consecutive in text), merge it
                    if current_entity and label == current_label and start == previous_end:
                        current_entity += word  # No space if directly consecutive
                    elif current_entity and label == current_label:
                        current_entity += " " + word  # Add space if it's part of the same entity
                    else:
                        # Store completed entity before starting a new one
                        if current_entity and current_label in ["PER", "ORG", "LOC", "EVN"]:
                            bert_entities.append(current_entity)

                        # Start a new entity if confidence is high
                        if score > 0.90 and label in ["PER", "ORG", "LOC", "EVN"]:
                            current_entity = word
                            current_label = label
                        else:
                            current_entity = ""
                            current_label = None

                previous_end = end  # Update last processed token end position

            # Store last entity
            if current_entity and current_label in ["PER", "ORG", "LOC", "EVN"]:
                bert_entities.append(current_entity)



            # Try to capture missing names using capitalized words
        
            all_entities_str = json.dumps(bert_entities, ensure_ascii=False) if bert_entities else ""

            
            # Write to CSV
            sentence = clean_sentence(sentence)
            metadata = [{"source": "SVT"}]
            row = [sentence, all_entities_str, metadata]
            writer.writerow(row)

            # Free memory
            elem.clear()

            count += 1
            if count >= 50:  # Limit to 50 sentences for testing
                break

print(f"✅ Processed first 500 sentences with BERT! Saved as {csv_file}.")