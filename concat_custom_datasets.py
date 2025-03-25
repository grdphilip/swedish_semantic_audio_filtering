import json

filepaths = [
    #("filtering/data/syndata_fb_train_manifest.json", "fb"),
    ("data/manifest_data/finetuning/preprocessed/syndata_11labs_train_manifest.json", "elevenlabs"),
    #("data/manifest_data/finetuning/preprocessed/common_voice_train_manifest.json", "common_voice"),
    ("data/manifest_data/finetuning/preprocessed/fleurs_train_manifest.json", "fleurs"),
    # ("data/manifest_data/finetuning/preprocessed/syndata_11labs_val_manifest.json", "elevenlabs"),
    # ("data/manifest_data/finetuning/preprocessed/common_voice_val_manifest.json", "common_voice"),
    # ("data/manifest_data/finetuning/preprocessed/fleurs_val_manifest.json", "fleurs")
    
]

def concat_datasets(filepaths):
    combined_train_manifest = []

    for filepath, source in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())  # Read each line as a JSON object
                    combined_train_manifest.append(data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {filepath}: {e}")
    
    return combined_train_manifest

# Example usage
combined_data = concat_datasets(filepaths)
print(f"Total entries: {len(combined_data)}")

# Save the combined data back in the same JSONL format (one JSON object per line)
output_filepath = "data/manifest_data/finetuning/preprocessed/"
combinations = "_".join([source for _, source in filepaths])
output_filepath += f"combined_{combinations}_train_manifest.jsonl"

with open(output_filepath, 'w', encoding='utf-8') as out_file:
    for item in combined_data:
        out_file.write(json.dumps(item) + '\n')  # Write each JSON object on a new line

print(f"Saved combined dataset to: {output_filepath}")

