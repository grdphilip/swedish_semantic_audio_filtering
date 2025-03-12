import json

filepaths = [
    ("filtering/data/syndata_fb_train_manifest.json", "fb"),
    ("filtering/data/syndata_11labs_train_manifest.json", "elevenlabs"),
    ("filtering/data/common_voice_train_manifest.json", "common_voice")
]

def concat_datasets(filepaths):
    
    # Concatenate datasets
    combined_train_manifest = []
    
    for filepath, source in filepaths:
        with open(filepath) as f:
            data = json.load(f)
            if source == "common_voice":
                data = data[:100]  # Only take 100 rows from common_voice dataset
            for entry in data:
                entry['source'] = source
            combined_train_manifest.extend(data)
    
    return combined_train_manifest

# Example usage
combined_data = concat_datasets(filepaths)

# Save the combined data to a JSON file
output_filepath = "filtering/data/combined_train_manifest.json"
with open(output_filepath, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)
