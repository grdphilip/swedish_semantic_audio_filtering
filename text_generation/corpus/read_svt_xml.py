import json
import csv
import xml.etree.ElementTree as ET
import re
from transformers import pipeline

# Load Swedish BERT NER model
nlp = pipeline("ner", model="KBLab/bert-base-swedish-cased-ner", tokenizer="KBLab/bert-base-swedish-cased-ner", device = 0)

# Define input/output files
xml_file = "svt-2023.xml"
csv_file = "../dataset/utterances.csv"

def clean_sentence(sentence: str) -> str:
    # Remove leading '-'
    sentence = re.sub(r'^-\s*', '', sentence)
    
    # Replace '–' with ','
    sentence = sentence.replace('–', ',')
    
    # Remove unnecessary quotation marks and commas
    sentence = sentence.replace('”', '').replace(',', '')
    
    # Replace ' : ' with ' ', but not when words are adjacent
    sentence = re.sub(r'(?<=\s):\s', ' ', sentence)
    
    # Remove occurrences of [ ... ]
    sentence = re.sub(r'\[.*?\]', '', sentence)
    
    # Replace middle quotation marks with a comma before the quoted word
    sentence = re.sub(r'(?<=\S)\s*["””]\s*(?=\S)', ',', sentence)
    
    return sentence

def group_entities(tokens):
    """
    Generalized entity grouping logic. Only merge tokens if:
    1. They are part of the same entity.
    2. The entity type does not change.
    3. Ensure subword tokens (starting with '##') are correctly merged only if they belong to the same entity type.
    """
    grouped_entities = []
    current_entity = ""
    current_label = None
    previous_end = -1  # Tracks end position of the last processed token

    for token in tokens:
        word = token["word"]
        label = token["entity"]
        score = token["score"]
        start = token["start"]
        end = token["end"]

        # Handle subword tokens (those starting with '##')
        if word.startswith("##"):
            # If it's a subword of the same entity, continue merging
            if current_label == label:
                current_entity += word[2:]  # Append without space
            else:
                # Treat subword as a new entity if it doesn't match the current entity type
                if current_entity and current_label in ["PER", "ORG", "LOC", "EVN"]:
                    grouped_entities.append(current_entity)
                current_entity = word[2:]  # Start new entity from subword
                current_label = label
        else:
            # If the entity changes or a new word starts a different entity, finalize the current entity
            if current_entity and current_label == label:
                current_entity += " " + word  # Continue adding the same entity
            elif current_entity:
                # Add the previous entity to the list before starting a new one
                if current_label in ["PER", "ORG", "LOC", "EVN"]:
                    grouped_entities.append(current_entity)
                current_entity = word  # Start new entity
                current_label = label
            else:
                # If no previous entity, start a new one
                current_entity = word
                current_label = label

        previous_end = end  # Update last processed token end position

    # Append the last entity if valid
    if current_entity and current_label in ["PER", "ORG", "LOC", "EVN"]:
        grouped_entities.append(current_entity)

    return grouped_entities


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

            # Get tokens and their entities from BERT NER
            tokens = nlp(sentence)

            # Process the tokens to handle subword grouping
            grouped_entities = group_entities(tokens)

            # Only write if there are entities to write
            if len(grouped_entities) > 0:
                all_entities_str = json.dumps(grouped_entities, ensure_ascii=False)
            else:
                continue

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

print(f"✅ Processed first 50 sentences with BERT! Saved as {csv_file}.")
