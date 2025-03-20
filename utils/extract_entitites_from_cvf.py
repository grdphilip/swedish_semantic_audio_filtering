from datasets import Dataset, load_dataset
from datasets import concatenate_datasets
from transformers import pipeline
from datasets import Audio, Features, Value, Sequence
from huggingface_hub import login
import os


# Load the Fleurs dataset and filter for Swedish (sv_se) and entity rec model
fleurs_df = load_dataset("google/fleurs", "sv_se")
common_voice_df = load_dataset("mozilla-foundation/common_voice_11_0", "sv-SE")
nlp = pipeline("ner", model="KBLab/bert-base-swedish-cased-ner", tokenizer="KBLab/bert-base-swedish-cased-ner", device = 0)

def group_entities(tokens):
    """
    Generalized entity grouping logic. Merge tokens if:
    1. They are part of the same entity (same label).
    2. They are contiguous by character positions or by token order.
    3. Subword tokens (starting with '##') are merged if they share the same label.
    4. Punctuation tokens that are allowed joiners (hyphens, dashes, colons) are merged without extra spaces.
    5. Tokens with a confidence score below 75% are ignored.
    6. A possessive marker "s" (or similar) is merged without a space if nearly contiguous.
    
    Additionally, if tokens are consecutive in the token list, a space is added (unless the previous character is an allowed joiner),
    but if tokens are nearly contiguous by character positions (gap <= 2) but not consecutive in order, they are merged without a space.
    """
    valid_labels = ["PER", "ORG", "LOC", "EVN", "OBJ", "LOC/PRS", "WRK"]
    # Allowed joiners to merge without adding an extra space.
    allowed_joiners = {"-", "–", "—", ":"}
    
    grouped_entities = []
    current_entity = ""
    current_label = None
    previous_end = -1      # End position of the last processed token
    previous_index = None  # Token index of the last processed token
    metadata = []
    
    for token in tokens:
        word = token["word"]
        label = token["entity"]
        start = token["start"]
        end = token["end"]
        index = token["index"]
        score = token["score"]
        
        # Ignore tokens with low confidence.
        if score < 0.0:
            if current_entity and current_label in valid_labels:
                grouped_entities.append(current_entity)
                metadata.append({"entity": current_entity, "entity_type": current_label})
            current_entity = ""
            current_label = None
            previous_end = end
            previous_index = None
            continue
        
        # Process punctuation tokens.
        if not any(c.isalnum() for c in word):
            if word in allowed_joiners:
                if current_entity:
                    # Merge joiner without a space.
                    current_entity += word
                else:
                    current_entity = word
                previous_end = end
                previous_index = index
                continue
            else:
                # For other punctuation, finish the current entity.
                if current_entity and current_label in valid_labels:
                    grouped_entities.append(current_entity)
                    metadata.append({"entity": current_entity, "entity_type": current_label})
                current_entity = ""
                current_label = None
                previous_end = end
                previous_index = None
                continue
        
        # Handle subword tokens (starting with '##').
        if word.startswith("##"):
            if current_label == label:
                current_entity += word[2:]
            else:
                if current_entity and current_label in valid_labels:
                    grouped_entities.append(current_entity)
                    metadata.append({"entity": current_entity, "entity_type": current_label})
                current_entity = word[2:]
                current_label = label
        else:
            # Merge possessive markers (like "s") without space if nearly contiguous.
            if current_entity and word.lower() == "s" and current_label == label and (start - previous_end) <= 2:
                current_entity += word
            # Merge if same label.
            elif current_entity and current_label == label:
                if previous_index is not None and index == previous_index + 1:
                    # Consecutive tokens in order: add space unless previous char is an allowed joiner.
                    if current_entity and current_entity[-1] in allowed_joiners:
                        current_entity += word
                    else:
                        current_entity += " " + word
                elif (start - previous_end) <= 2:
                    # Nearly contiguous by character positions but not consecutive: merge without space.
                    current_entity += word
                else:
                    # Not contiguous: finish the current entity and start a new one.
                    if current_label in valid_labels:
                        grouped_entities.append(current_entity)
                        metadata.append({"entity": current_entity, "entity_type": current_label})
                    current_entity = word
                    current_label = label
            else:
                # New entity.
                if current_entity and current_label in valid_labels:
                    grouped_entities.append(current_entity)
                    metadata.append({"entity": current_entity, "entity_type": current_label})
                current_entity = word
                current_label = label
        
        previous_end = end
        previous_index = index
    
    if current_entity and current_label in valid_labels:
        grouped_entities.append(current_entity)
        metadata.append({"entity": current_entity, "entity_type": current_label})
    
    return grouped_entities, metadata



def create_entity_dataset(dataset):
    
    dataset_with_entities = []
    
    for i in dataset:
        tokens = nlp(i['text'])
        grouped_entities, metadata = group_entities(tokens)
        
        if len(grouped_entities) > 0:
            i['entities'] = grouped_entities
            i['metadata'] = metadata
            dataset_with_entities.append(i)
            
    return dataset_with_entities
        


fleurs_train = fleurs_df['train']
fleurs_val = fleurs_df['validation']
fleurs_test = fleurs_df['test']

common_voice_train = common_voice_df['train']
common_voice_val = common_voice_df['validation']
common_voice_test = common_voice_df['test']


cv_concat = concatenate_datasets([common_voice_train, common_voice_val, common_voice_test])
fleurs_concat = concatenate_datasets([fleurs_train, fleurs_val, fleurs_test])

# print(len(cv_concat)) #17429 #sentence, audio, path, entities
# print(len(fleurs_concat)) #3474

# CV colnames ['client_id','up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment']
# Fluers colnames ['id', 'num_samples', 'transcription',  'gender', 'lang_id', 'language', 'lang_group_id']
# print(cv_concat.column_names) 
# print(fleurs_concat.column_names)

cv_range = len(cv_concat) - 1
fleurs_range = len(fleurs_concat) - 1

# Behåll audio, path, text, entities, metadata: [{entity, entity_type}]
cv_concat_subset_rows = cv_concat.select(range(cv_range)).map(lambda example: {
    'audio': example['audio'],
    'path': example['path'],
    'text': example['sentence'],
    'entities': [],
    'metadata': []
}, remove_columns=['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment', 'sentence'])

#fleurs_concat_subset_rows = fleurs_concat.select(range(fleurs_range)).map(lambda example: {
#    'audio': example['audio'],
#    'path': example['path'],
#    'text': example['raw_transcription'],
#    'entities': [],
#    'metadata': []
#}, remove_columns=['id', 'num_samples', 'transcription', 'gender', 'lang_id', 'language', 'lang_group_id', 'raw_transcription'])

# print(cv_concat_subset_first_100_rows.column_names)
# print(fleurs_concat_subset_first_100_rows.column_names)

    
cv_with_entities = create_entity_dataset(cv_concat_subset_rows)


for i in cv_with_entities:
    print(f"entities: {i['entities']}")
    
    
#fleurs_with_entities = create_entity_dataset(fleurs_concat_subset_rows)


#for i in fleurs_with_entities:
#    print(f"text: {i['text']}, entities: {i['entities']}")

print(len(cv_concat))
#print(len(fleurs_concat))
print(len(cv_with_entities))
#print(len(fleurs_with_entities))


#fleurs_features = Features({
#    'audio': Audio(sampling_rate=16000),
#    'path': Value('string'),
#    'text': Value('string'),
#    'entities': Sequence(Value('string')),
#    'metadata': Sequence({
#        'entity': Value('string'),
#        'entity_type': Value('string')
#    })
#})

cv_features = Features({
    'audio': Audio(sampling_rate=16000),
    'path': Value('string'),
    'text': Value('string'),
    'entities': Sequence(Value('string')),
    'metadata': Sequence({
        'entity': Value('string'),
        'entity_type': Value('string')
    })
})

#fleurs_manifest_dict = {
#    'audio': [item['audio'] for item in fleurs_with_entities],
#    'path': [item['path'] for item in fleurs_with_entities],
#    'text': [item['text'] for item in fleurs_with_entities],
#    'entities': [item['entities'] for item in fleurs_with_entities],
#    'metadata': [item['metadata'] for item in fleurs_with_entities]
#}

cv_manifest_dict = {
    'audio': [item['audio'] for item in cv_with_entities],
    'path': [item['path'] for item in cv_with_entities],
    'text': [item['text'] for item in cv_with_entities],
    'entities': [item['entities'] for item in cv_with_entities],
    'metadata': [item['metadata'] for item in cv_with_entities]
}



# ================ UPLOAD TO HUGGING FACE DATASETS ================

token = ""
os.environ['HF_TOKEN'] = token
login(token=token)

#fleurs_dataset = Dataset.from_dict(fleurs_manifest_dict, features=fleurs_features)
#fleurs_dataset.push_to_hub("grdphilip/fleurs_swedish_with_entities")

cv_dataset = Dataset.from_dict(cv_manifest_dict, features=cv_features)
cv_dataset.push_to_hub("grdphilip/cv_swedish_with_entities")
    