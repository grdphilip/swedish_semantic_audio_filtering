import json

filepaths = [
    #("filtering/data/syndata_fb_train_manifest.json", "fb"),
    ("data/manifest_data/finetuning/preprocessed/syndata_11labs_train_manifest.json", "elevenlabs"),
    #("filtering/data/common_voice_train_manifest.json", "common_voice"),
    ("data/manifest_data/finetuning/preprocessed/fleurs_train_manifest.json", "fleurs"),
]

def concat_datasets(filepaths):
    # Concatenate datasets
    combined_train_manifest = []
    
    for filepath, source in filepaths:
        with open(filepath) as f:
            try:
                # If file has multiple JSON objects, handle it line by line
                for line in f:
                    data = json.loads(line)  # Read each JSON object from a line
                    combined_train_manifest.append(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {filepath}: {e}")
                continue
    
    return combined_train_manifest

# Example usage
combined_data = concat_datasets(filepaths)
print(len(combined_data))

# Save the combined data to a JSON file
output_filepath = "data/manifest_data/finetuning/preprocessed/"
combinations = "_".join([source for _, source in filepaths])
print(combinations)
output_filepath += f"combined_{combinations}_train_manifest.json"


# Save to file
with open(output_filepath, 'w') as out_file:
    json.dump(combined_data, out_file, indent=4)


