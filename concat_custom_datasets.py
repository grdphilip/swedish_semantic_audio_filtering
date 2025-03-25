import json

filepaths = [
    ("filtering/data/syndata_fb_train_manifest.json", "fb"),
    ("filtering/data/syndata_11labs_train_manifest.json", "elevenlabs"),
    ("filtering/data/common_voice_train_manifest.json", "common_voice"),
    ("filtering/data/fleurs_train_manifest.json", "fleurs"),
]

def concat_datasets(filepaths):
    
    # Concatenate datasets
    combined_train_manifest = []
    
    for filepath, source in filepaths:
        with open(filepath) as f:
            data = json.load(f)
            combined_train_manifest.extend(data)
    
    return combined_train_manifest

# Example usage
combined_data = concat_datasets(filepaths)

# Save the combined data to a JSON file

output_filepath = "filtering/data/"
combinations = "_".join([source for _, source in filepaths])
output_filepath += f"combined_{combinations}_train_manifest.json"
with open(output_filepath, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)
